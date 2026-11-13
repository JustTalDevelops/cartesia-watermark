"""Thin Cartesia API client: TTS generation and watermark detection."""
import json
import time
import os
import pathlib
import sys
import fcntl
from email.utils import parsedate_to_datetime

import requests

API = "https://api.cartesia.ai"
VERSION = "2026-08-14"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _env(name):
    if name in os.environ:                               # explicit env wins, even if empty
        return os.environ[name] or None
    v = None
    if (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(name + "="):
                v = line.split("=", 1)[1].strip()
    return v


def _headers():
    return {"X-API-Key": _env("CARTESIA_API_KEY"), "Cartesia-Version": VERSION}


class RateLimited(RuntimeError):
    pass


class EgressChanged(RuntimeError):
    pass


_pinned_ip = [None]


def _egress_ip(proxies):
    return requests.get("https://api.ipify.org", proxies=proxies, timeout=30).text.strip()


def tts(transcript, out, model="sonic-3.6", voice="47c38ca4-5f35-497b-b1a3-415245fb35e1",
        container="wav", encoding="pcm_f32le", sample_rate=44100, bit_rate=None, **extra):
    fmt = {"container": container, "sample_rate": sample_rate}
    if container != "mp3":
        fmt["encoding"] = encoding
    else:
        fmt["bit_rate"] = bit_rate or 192000
    body = {"model_id": model, "transcript": transcript, "voice": {"mode": "id", "id": voice},
            "output_format": fmt, "language": "en", **extra}
    r = requests.post(f"{API}/tts/bytes", headers=_headers(), json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"tts {r.status_code}: {r.text[:500]}")
    pathlib.Path(out).write_bytes(r.content)
    return dict(r.headers)


_last = [0.0]
MIN_INTERVAL = float(_env("DETECT_INTERVAL") or 10)


def _pace_detector(cooldown=False, cooldown_seconds=300):
    """Share pacing and 429 cooldowns across concurrent experiment processes."""
    directory = ROOT / ".cache"
    directory.mkdir(exist_ok=True)
    state_path = directory / "detector_pacing.json"
    state_path.touch(exist_ok=True)
    while True:
        with state_path.open("r+") as state_file:
            fcntl.flock(state_file, fcntl.LOCK_EX)
            state_file.seek(0)
            try:
                state = json.load(state_file)
            except (ValueError, OSError):
                state = {}
            now = time.time()
            if cooldown:
                state["cooldown_until"] = max(state.get("cooldown_until", 0), now + cooldown_seconds)
                wait = 0
            else:
                if state.get("cooldown_until", 0) > now:
                    raise RateLimited("Detector is in a saved 429 cooldown; no request sent")
                interval = max(MIN_INTERVAL, state.get("interval", MIN_INTERVAL))
                wait = max(0, state.get("last_start", 0) + interval - now)
                if not wait:
                    state.update(last_start=now, interval=MIN_INTERVAL)
            if not wait:
                state_file.seek(0)
                json.dump(state, state_file)
                state_file.truncate()
                state_file.flush()
                return
        # Release the lock while waiting so a 429 can cancel queued requests.
        time.sleep(min(wait, 30))


def detect(path):
    """Query the detector. Paced; raises RateLimited on 429 instead of retrying."""
    proxy = _env("DETECT_PROXY")
    proxies = {"https": proxy, "http": proxy} if proxy else None
    if proxies:                                          # stay on one IP: stop if the provider re-assigns it
        ip = _egress_ip(proxies)
        if _pinned_ip[0] is None:
            _pinned_ip[0] = ip
        elif ip != _pinned_ip[0]:
            raise EgressChanged(f"proxy egress moved {_pinned_ip[0]} -> {ip}; stopping instead of rotating")
    # Spaced starts can still overlap when inference takes longer than the
    # interval. Serialize the entire HTTP request across experiment processes.
    request_lock = ROOT / ".cache/detector_request.lock"
    request_lock.parent.mkdir(exist_ok=True)
    with request_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(10):                        # retry on the same connection route
            try:
                _pace_detector()
                with open(path, "rb") as f:
                    r = requests.post(f"{API}/v1/watermark/detect", headers=_headers(), proxies=proxies,
                                      files={"file": (pathlib.Path(path).name, f)}, timeout=120)
                break
            except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError):
                if attempt == 9:
                    raise
                time.sleep(30)
        if r.status_code == 429:
            try:
                retry_after = float(r.headers.get("Retry-After", 300))
            except (TypeError, ValueError):
                try:
                    retry_after = parsedate_to_datetime(r.headers['Retry-After']).timestamp()-time.time()
                except (TypeError, ValueError, KeyError, OverflowError):
                    retry_after = 300
            allowed = {'retry-after','x-ratelimit-limit','x-ratelimit-remaining','x-ratelimit-reset',
                       'ratelimit-limit','ratelimit-remaining','ratelimit-reset','server','date'}
            (request_lock.parent/'detector_last_429.json').write_text(json.dumps({
                'time':time.time(), 'headers':{k:v for k,v in r.headers.items() if k.lower() in allowed},
                'cooldown_seconds':max(300,retry_after)},indent=2))
            _pace_detector(cooldown=True, cooldown_seconds=max(300, retry_after))
            raise RateLimited(f"429 from {r.headers.get('server')} on {path}")
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:500]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, *detect(p))
