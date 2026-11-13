"""Quality-constrained, per-file refinement of an already trained remover.

Experimental probe; optimization uses the two frozen local detector families.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["FEAT"] = "xlsr2"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from train_remover import Teacher, fidelity
from remover import ROOT, load_model, resample, quality_metrics
from eval_remover import speech_metrics
import cartesia


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/remover_wave3local_probe200.pt")
    parser.add_argument("--tag", default="refine1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--local-snr-db", type=float, default=27)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--intelligibility-weight", type=float, default=200)
    parser.add_argument("--spectral-weight", type=float, default=10)
    parser.add_argument("--pesq-weight", type=float, default=20)
    parser.add_argument("--teacher-config", type=Path, default=ROOT / "results/final_detector_r4.json")
    parser.add_argument("--min-snr-db", type=float)
    parser.add_argument("--raw-teacher", type=Path)
    parser.add_argument("--raw-weight", type=float, default=2.0)
    parser.add_argument("--sources", nargs='+', help='Development source basenames to refine, in this order')
    parser.add_argument("--min-pesq", type=float, default=4.3)
    parser.add_argument("--min-estoi", type=float, default=.985)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(2029)
    model, checkpoint = load_model(args.checkpoint)
    model.config["local_snr_db"] = args.local_snr_db
    if args.min_snr_db is not None:
        model.config["min_snr_db"] = args.min_snr_db
    model.requires_grad_(False)
    teacher = Teacher(cross_detector=True, margin=4, spectral_weight=args.spectral_weight,
                      intelligibility_weight=args.intelligibility_weight, pesq_weight=args.pesq_weight,
                      config=args.teacher_config)
    raw_teacher = None
    if args.raw_teacher:
        from raw_oracle import load_raw_oracle
        raw_teacher = load_raw_oracle(args.raw_teacher)[0].requires_grad_(False)
    out = ROOT / "audio" / args.tag
    out.mkdir(exist_ok=True)
    ledger = ROOT / "results" / f"remover_{args.tag}_cartesia.json"
    reference = json.loads((ROOT / "results/remover_wave1_probe250_cartesia.json").read_text())
    if args.sources:
        by_name = {Path(r['source']).name:r for r in reference['clips']}
        reference['clips'] = [by_name[name] for name in args.sources]
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "algorithm": "band-limited projected Adam on XLS-R + WavLM", "steps": args.steps,
              "local_snr_db": args.local_snr_db, "clips": []}
    report.update(teacher_config=teacher.detector_config, teacher_member_sha256=teacher.member_hashes)
    if args.raw_teacher:
        report.update(raw_teacher_sha256=hashlib.sha256(args.raw_teacher.read_bytes()).hexdigest(),
                      raw_teacher=str(args.raw_teacher), raw_weight=args.raw_weight)
    started = time.time()
    api_rate_limited = False
    for source in reference["clips"]:
        path = ROOT / source["source"]
        x, sr = sf.read(path)
        x16 = resample(x, sr, 16000)
        scale = max(1e-5, float(np.sqrt(np.mean(x16 ** 2))))
        original = torch.tensor(x16 / scale, dtype=torch.float32)[None]
        with torch.no_grad():
            initial = model(original) - original
            if teacher.intelligibility is not None:
                for _ in range(7):
                    if float(-teacher.intelligibility(original + initial, original)[0]) >= 0.995:
                        break
                    initial *= 0.5
        delta = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([delta], lr=0.003)
        history = []
        best, best_value, best_row = None, float("inf"), None
        for step in range(args.steps + 1):
            filtered = F.conv1d(delta[:, None], model.bandpass_fir, padding=64)[:, 0]
            edited = model._project(original, filtered)
            a, b = teacher(edited), teacher.cross_logits(edited)
            mse, spectral = fidelity(original, edited)
            loss = (F.softplus(a + teacher.margin).mean() + 0.5 * F.softplus(b + teacher.margin).mean()
                    + 100 * mse.mean() + teacher.spectral_weight * spectral.mean())
            raw = raw_teacher(edited) if raw_teacher is not None else None
            if raw is not None:
                loss = loss + args.raw_weight * F.softplus(raw + 3).mean()
            if teacher.intelligibility is not None:
                proxy_estoi = -teacher.intelligibility(edited, original)
                loss = loss + teacher.intelligibility_weight * F.relu(0.995 - proxy_estoi).mean()
            if teacher.perceptual is not None:
                loss = loss + teacher.pesq_weight * F.relu(4.4 - teacher.perceptual.mos(original, edited)).mean()
            if step % 5 == 0 or step == args.steps:
                difference = (edited - original).detach()[0].numpy().astype(np.float64) * scale
                native = resample(difference, 16000, sr)[:len(x)]
                native = np.pad(native, (0, max(0, len(x) - len(native))))
                y = x + native
                metrics = {**quality_metrics(x, y), **speech_metrics(x, y, sr)}
                row = {"step": step, "xlsr": float(a.detach()[0]), "wavlm": float(b.detach()[0]), **metrics}
                if raw is not None:
                    row['raw_oracle'] = float(raw.detach()[0])
                history.append(row)
                print(path.name, json.dumps(row), flush=True)
                value = float(a.detach()[0] + 0.5 * b.detach()[0])
                if raw is not None:
                    value += args.raw_weight * float(raw.detach()[0])
                if metrics["pesq_wb"] >= args.min_pesq and metrics["estoi"] >= args.min_estoi and value < best_value:
                    best, best_value = y.copy(), value
                    best_row = row.copy()
            if step == args.steps:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.copy_(model._project(original, delta) - original)
        if best is None:
            print("No candidate passed the quality gate", path.name, flush=True)
            continue
        output = out / path.name
        sf.write(output, best, sr, subtype="FLOAT")
        record = {"source": str(path.relative_to(ROOT)), "output": str(output.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "original": source["original"], "original_cached_from": "remover_wave1_probe250_cartesia.json",
            "history": history, **quality_metrics(x, best), **speech_metrics(x, best, sr)}
        record["selected"] = best_row
        report["clips"].append(record)
        ledger.write_text(json.dumps(report, indent=2))
        if args.verify and not api_rate_limited:
            try:
                code, result = cartesia.detect(output)
                if code == 200 and isinstance(result.get("detected"), bool):
                    record["edited"] = result["detected"]
                    print(path.name, "Cartesia detected=", record["edited"], flush=True)
            except cartesia.RateLimited:
                print("Cartesia rate limited; keeping the local measurements", flush=True)
                api_rate_limited = True
            except Exception as error:
                print(f"Cartesia request failed ({type(error).__name__}); keeping local measurements", flush=True)
            ledger.write_text(json.dumps(report, indent=2))
    print(f"Finished in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
