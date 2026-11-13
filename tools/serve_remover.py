"""Private upload / playback / download UI for local speech processing.

    .venv/bin/python tools/serve_remover.py --host 127.0.0.1 --port 7860

Use Tailscale Serve or bind explicitly to the machine's Tailscale IP. The
service never calls Cartesia. Uploaded files expire after 24 hours.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import soundfile as sf
import torch
import uvicorn

from remover import ROOT, load_model, process_audio
from reconstruction_worker import ReconstructionWorker


LOGGER = logging.getLogger("recording_tester")
WEB = ROOT / "web"
DATA = ROOT / ".cache" / "recording_tester"
CHECKPOINT = ROOT / "results/remover_quality_v3.pt"
CHECKPOINT_SHA256 = "c5ddec58926c24e0f92c83475d589286ea98f2a8bc95124097ef567cc1910f8b"
RECONSTRUCTION_PYTHON = ROOT / '.cache/resemble-venv/bin/python'
RECONSTRUCTION_SCRIPT = ROOT / 'tools/reconstruct_audio.py'
reconstructor = ReconstructionWorker(RECONSTRUCTION_PYTHON, RECONSTRUCTION_SCRIPT)
RECONSTRUCTION_METHODS = {'reconstruct': 'standard', 'reconstruct_strong': 'strong'}
MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 15 * 60
RETENTION = 24 * 60 * 60
MAX_JOBS = 20
MAX_PENDING = 3
FORMATS = "wav,mp3,flac,ogg,mov,matroska,webm,aac,aiff,au"
jobs = {}
lock = threading.RLock()
slots = threading.BoundedSemaphore(MAX_PENDING)
model = None
executor = None


def update(job_id, **fields):
    with lock:
        jobs[job_id].update(fields)


def sweep():
    """Bound retained uploads and remove abandoned uploads after a restart."""
    now = time.time()
    with lock:
        expired = [key for key, row in jobs.items()
                   if row["state"] in ("done", "error") and now - row["created"] > RETENTION]
        for key in expired:
            shutil.rmtree(DATA / key, ignore_errors=True)
            jobs.pop(key, None)
        for directory in DATA.iterdir():
            if directory.is_dir() and directory.name not in jobs and now - directory.stat().st_mtime > RETENTION:
                shutil.rmtree(directory, ignore_errors=True)


def public_job(row):
    return {key: value for key, value in row.items() if key not in ("created",)}


def checked_job(job_id):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "This recording is no longer available. Upload it again.")
    with lock:
        row = jobs.get(job_id)
        if not row:
            raise HTTPException(404, "This recording is no longer available. Upload it again.")
        return dict(row)


def run_media(command):
    try:
        return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                              check=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise ValueError("This recording took too long to decode. Try a shorter clip.") from None
    except subprocess.CalledProcessError:
        raise ValueError("This file could not be read as audio. Try a WAV, MP3, M4A or FLAC file.") from None


def process_job(job_id):
    directory = DATA / job_id
    try:
        update(job_id, state="processing", message="Reading your recording…")
        probe = run_media(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                           "-format_whitelist", FORMATS, "-select_streams", "a:0",
                           "-show_entries", "stream=sample_rate,channels:format=duration", "-of", "json",
                           str(directory / "upload")])
        details = json.loads(probe.stdout)
        if not details.get("streams"):
            raise ValueError("There is no audio track in this file. Choose another recording.")
        stream = details["streams"][0]
        sample_rate, channels = int(stream["sample_rate"]), int(stream["channels"])
        if not 8000 <= sample_rate <= 96000 or channels not in (1, 2):
            raise ValueError("Choose a mono or stereo recording sampled between 8 and 96 kHz.")
        duration = float(details.get("format", {}).get("duration", 0))
        if not math.isfinite(duration) or duration > MAX_SECONDS:
            raise ValueError("Choose a recording shorter than 15 minutes.")
        reference = directory / "reference.wav"
        run_media(["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
                   "-format_whitelist", FORMATS, "-i", str(directory / "upload"),
                   "-map", "0:a:0", "-vn", "-t", str(MAX_SECONDS + 1),
                   "-c:a", "pcm_f32le", str(reference)])
        audio, actual_rate = sf.read(reference)
        if not len(audio) or len(audio) / actual_rate > MAX_SECONDS or not np.isfinite(audio).all():
            raise ValueError("Choose a valid recording between a moment and 15 minutes long.")
        update(job_id, message="Processing your recording…", seconds=len(audio) / actual_rate,
               sample_rate=actual_rate, channels=channels)
        method = jobs[job_id].get('method', 'editor')
        if method in RECONSTRUCTION_METHODS:
            update(job_id, message='Reconstructing speech… This takes about 2–3 minutes per minute of audio.')
            try:
                reconstructor.run(reference, directory/'processed.wav', RECONSTRUCTION_METHODS[method],
                                  directory/'reconstruction.json',
                                  timeout=max(180, math.ceil(len(audio)/actual_rate*8+120)))
            except subprocess.TimeoutExpired:
                raise ValueError('Reconstruction took too long. Try a shorter recording.') from None
            except (RuntimeError, OSError, ValueError):
                raise ValueError('Reconstruction could not process this recording. Try Small edits or a shorter file.') from None
            edited, result_rate = sf.read(directory/'processed.wav')
            if result_rate != actual_rate:
                raise ValueError('Reconstruction could not preserve the sample rate.')
        else:
            edited = process_audio(model, audio, actual_rate, strength=jobs[job_id]["strength"])
        if edited.shape != audio.shape or not np.isfinite(edited).all():
            raise ValueError("Processing could not preserve this recording. Try another file.")
        update(job_id, message="Preparing your audio…")
        if method == 'editor':
            sf.write(directory / "processed.wav", edited, actual_rate, subtype="FLOAT")
        # Equal PCM playback encoding keeps the A/B comparison browser-friendly.
        # The downloadable result above retains the model's float precision.
        sf.write(directory / "original_preview.wav", audio, actual_rate, subtype="PCM_16")
        sf.write(directory / "processed_preview.wav", edited, actual_rate, subtype="PCM_16")
        update(job_id, state="done", message="Your audio is ready.",
               elapsed_seconds=time.time() - jobs[job_id]["created"])
        (directory / "job.json").write_text(json.dumps(jobs[job_id], indent=2))
    except Exception as error:
        if isinstance(error, ValueError):
            message = str(error)
        else:
            LOGGER.error("Processing failed: %s", type(error).__name__)
            message = "This recording could not be processed. Try a shorter file or a WAV export."
        update(job_id, state="error", message=message)
        shutil.rmtree(directory, ignore_errors=True)
    finally:
        slots.release()


@asynccontextmanager
async def lifespan(app):
    global model, executor
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be available on the service PATH")
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("The frozen V3 checkpoint changed")
    if not RECONSTRUCTION_PYTHON.is_file() or not RECONSTRUCTION_SCRIPT.is_file():
        raise RuntimeError('The local reconstruction runtime is unavailable')
    DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
    torch.set_num_threads(2)
    model, _ = load_model(CHECKPOINT)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio_editor")
    for path in DATA.glob("*/job.json"):
        try:
            row = json.loads(path.read_text())
            if (row["state"] == "done" and path.parent.name == row["id"]
                    and re.fullmatch(r"[0-9a-f]{32}", row["id"])):
                jobs[row["id"]] = row
        except (OSError, ValueError, KeyError, TypeError):
            continue
    sweep()

    async def housekeeping():
        while True:
            await asyncio.sleep(60)
            await asyncio.to_thread(sweep)

    cleaner = asyncio.create_task(housekeeping())
    try:
        yield
    finally:
        cleaner.cancel()
        executor.shutdown(wait=True)
        reconstructor.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def private_requests(request, call_next):
    if request.method in ("POST", "DELETE") and request.headers.get("x-requested-with") != "recording-tester":
        return JSONResponse({"detail": "Open the recording tester to upload or remove a file."}, status_code=403)
    if request.method == "POST":
        try:
            size = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "Invalid upload size."}, status_code=400)
        if size > MAX_BYTES:
            return JSONResponse({"detail": "This file is over 100 MB. Choose a smaller recording."}, status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    return response


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/app.js")
def script():
    return FileResponse(WEB / "app.js", media_type="text/javascript")


@app.get("/style.css")
def styles():
    return FileResponse(WEB / "style.css", media_type="text/css")


@app.get("/api/health")
def health():
    return {"ready": model is not None, "model": "Reconstruction", "default_method": "reconstruct", "max_upload_bytes": MAX_BYTES,
            "max_seconds": MAX_SECONDS, "retention_hours": RETENTION // 3600}


@app.post("/api/jobs", status_code=202)
async def upload(request: Request, filename: str = "recording", strength: float = 1, method: str = 'reconstruct'):
    if method not in (*RECONSTRUCTION_METHODS, 'editor'):
        raise HTTPException(422, 'Choose Reconstruction or Small edits.')
    if not math.isfinite(strength) or not 0.25 <= strength <= 1:
        raise HTTPException(422, "Edit amount must be between 25% and 100%.")
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "Three recordings are already being processed. Try again in a moment.")
    job_id = uuid.uuid4().hex
    directory = DATA / job_id
    submitted = False
    try:
        sweep()
        with lock:
            completed = sorted((r for r in jobs.values() if r["state"] in ("done", "error")),
                               key=lambda r: r["created"])
            while len(jobs) >= MAX_JOBS and completed:
                old = completed.pop(0)
                shutil.rmtree(DATA / old["id"], ignore_errors=True)
                jobs.pop(old["id"], None)
        directory.mkdir(mode=0o700)
        size = 0
        with (directory / "upload").open("xb") as target:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "This file is over 100 MB. Choose a smaller recording.")
                target.write(chunk)
        if not size:
            raise HTTPException(400, "This file is empty. Choose another recording.")
        name = re.split(r"[/\\]", filename)[-1]
        name = "".join(c for c in name if c.isprintable())[:180] or "recording"
        row = {"id": job_id, "filename": name, "strength": strength if method == 'editor' else 1,
               "method": method, "created": time.time(),
               "state": "queued", "message": "Your recording is queued."}
        with lock:
            jobs[job_id] = row
        executor.submit(process_job, job_id)
        submitted = True
        return {"id": job_id}
    finally:
        if not submitted:
            shutil.rmtree(directory, ignore_errors=True)
            with lock:
                jobs.pop(job_id, None)
            slots.release()


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    row = public_job(checked_job(job_id))
    if row['state'] == 'done':
        # Display only actual checker responses bound to this exact output.
        # Reading saved evidence does not send recordings anywhere.
        ledger = ROOT / '.cache/recording_tester_cartesia_checks.json'
        try:
            results = json.loads(ledger.read_text()).get('clips', [])
            matching_job = [r for r in results if r.get('job') == job_id and isinstance(r.get('detected'), bool)]
            if matching_job:
                current_hash = hashlib.sha256((DATA/job_id/'processed.wav').read_bytes()).hexdigest()
                checked = next((r for r in reversed(matching_job) if r.get('sha256') == current_hash), None)
                if checked:
                    row['verification'] = {'provider': 'Cartesia', 'detected': checked['detected'],
                                           'checked_at': checked['checked_at']}
        except (OSError, ValueError, TypeError):
            pass
        trial = listening_trial(job_id)
        if trial:
            row['trial'] = trial
    return row


def listening_trial(job_id):
    """Expose this research trial only when its exact full WAV checked negative.

    This does not alter the editor or trigger an external request. Trials remain
    inside the expiring job and are deleted with the original recording.
    """
    try:
        specification = DATA / job_id / 'research/listening_trial.json'
        spec = json.loads(specification.read_text()) if specification.is_file() else {}
        variant = spec.get('variant', 'generative_restoration32')
        if not re.fullmatch(r'[a-zA-Z0-9_]+', variant):
            return None
        path = DATA / job_id / 'research' / (variant + '.wav')
        preview = path.with_name(variant + '_preview.wav')
        if not path.is_file() or not preview.is_file():
            return None
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if spec and sha != spec.get('sha256'):
            return None
        results = json.loads((ROOT / '.cache/recording_tester_cartesia_checks.json').read_text())['clips']
        checked = next((r for r in reversed(results) if r.get('job') == job_id
                        and r.get('sha256') == sha and isinstance(r.get('detected'), bool)), None)
        review_name = ('recording_tester_generative_voice_review.json' if variant == 'generative_restoration32'
                       else f'recording_tester_{variant}_voice_review.json')
        reviews = json.loads((ROOT / '.cache' / review_name).read_text())
        reviewed = any(r.get('job') == job_id and r.get('output_sha256') == sha for r in reviews)
        if (checked and reviewed and checked['detected'] is False and
                checked.get('role') == spec.get('role', 'generative_restoration32_research_outside_V3_gates')):
            return {'detected': False, 'checked_at': checked['checked_at'],
                    'variant': variant,
                    'quality': spec.get('quality', 'Listener-approved reconstruction reference.')}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


@app.get("/api/jobs/{job_id}/audio/{kind}")
def recording(job_id: str, kind: str):
    row = checked_job(job_id)
    if row["state"] != "done":
        raise HTTPException(409, "This recording is not ready yet.")
    files = {"original": "original_preview.wav", "processed": "processed_preview.wav", "download": "processed.wav"}
    if kind in ('trial', 'trial-preview'):
        trial = listening_trial(job_id)
        if not trial:
            raise HTTPException(404, 'No verified listening trial is available for this recording.')
        files['trial'] = f'research/{trial["variant"]}.wav'
        files['trial-preview'] = f'research/{trial["variant"]}_preview.wav'
    if kind not in files:
        raise HTTPException(404, "Audio file not found.")
    filename = None
    if kind in ("download", "trial"):
        stem = Path(row["filename"]).stem or "recording"
        filename = stem + ("_reconstruction_trial.wav" if kind == 'trial' else "_processed.wav")
    return FileResponse(DATA / job_id / files[kind], media_type="audio/wav", filename=filename)


@app.delete("/api/jobs/{job_id}")
def remove(job_id: str):
    row = checked_job(job_id)
    if row["state"] not in ("done", "error"):
        raise HTTPException(409, "Wait for processing to finish before removing this recording.")
    with lock:
        jobs.pop(job_id, None)
        shutil.rmtree(DATA / job_id, ignore_errors=True)
    return {"removed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)
