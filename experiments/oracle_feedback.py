"""Collect real detector feedback on edited TRAINING voices for surrogate repair.

One fixed voice/text per training language, with both marked renders joined
without silence padding. Validation and sealed4 voices are excluded.
"""
import os
os.environ["FEAT"] = "xlsr2"
import hashlib
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import numpy as np
import soundfile as sf
import torch

import cartesia
import frames
from remover import ROOT, load_model, process_audio, quality_metrics
from eval_remover import speech_metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    output_dir = ROOT / "audio/oracle_round1"
    output_dir.mkdir(exist_ok=True)
    feature_dir = ROOT / "features_oracle/round1"
    feature_dir.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / "results/remover_oracle_round1_cartesia.json"
    manifest = json.loads((ROOT / "results/remover_wave2c.manifest.json").read_text())
    excluded = {r["voice"] for r in manifest["validation"]}
    models = {}
    for name, path in (("global", ROOT / "results/remover_wave1.pt"),
                       ("quality", ROOT / "results/remover_oracle_wave5.pt")):
        models[name] = load_model(path)[0]
    report = json.loads(ledger.read_text()) if ledger.exists() else {
        "api_version": cartesia.VERSION, "purpose": "training-only oracle feedback",
        "models": {"global": digest(ROOT / "results/remover_wave1.pt"),
                   "quality": digest(ROOT / "results/remover_oracle_wave5.pt")}, "clips": []}
    if not args.prepare_only and time.time() < report.get("next_allowed_at", 0):
        print("Cooling down after the saved rate limit; no requests sent", flush=True)
        return 2
    pools = {}
    for row in manifest["train"]:
        if row["marked"]:
            pools.setdefault(row["language"], []).append(row)
    rng = random.Random(2030)
    used, plan = set(), []
    for language, rows in sorted(pools.items()):
        rows = rows.copy(); rng.shuffle(rows)
        for row in rows:
            if row["voice"] in used or row["voice"] in excluded:
                continue
            source = ROOT / row["path"]
            prefix = source.stem.rsplit("_", 1)[0]
            pair = [source.with_name(f"{prefix}_{m}.wav") for m in ("sonic-3.6", "sonic-preview")]
            if not all(p.exists() for p in pair):
                continue
            a, sr = sf.read(pair[0]); b, sr2 = sf.read(pair[1])
            if sr != sr2 or (len(a)+len(b))/sr < 4.7:
                continue
            source_path = output_dir / f"{language}_source.wav"
            if not source_path.exists():
                sf.write(source_path, np.concatenate([a, np.zeros(int(0.3*sr)), b]), sr, subtype="FLOAT")
            plan.append({"source": str(source_path.relative_to(ROOT)), "voice": row["voice"],
                "language": language, "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in pair]})
            used.add(row["voice"])
            break
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    for item in plan:
        source = ROOT / item["source"]
        x, sr = sf.read(source)
        source_hash = digest(source)
        source_labels = report.setdefault("source_labels", {})
        if not args.prepare_only and source_hash not in source_labels:
            try:
                code, response = cartesia.detect(source)
            except cartesia.RateLimited:
                report["next_allowed_at"] = time.time()+300
                ledger.write_text(json.dumps(report, indent=2))
                print("Rate limited; saved progress and a five-minute cooldown", flush=True)
                return 2
            except Exception as error:
                print(f"Source request failed ({type(error).__name__}); progress saved", flush=True)
                return 2
            if code != 200 or not isinstance(response, dict) or not isinstance(response.get("detected"), bool):
                print(f"Unexpected source HTTP {code}; progress saved", flush=True)
                return 2
            source_labels[source_hash] = response["detected"]
            ledger.write_text(json.dumps(report, indent=2))
            print(f"{item['language']}_original: Cartesia={response['detected']}", flush=True)
        original_features = feature_dir / f"{item['language']}_original.npz"
        if not original_features.exists():
            h, rms = frames.frames(frames.to16k(x, sr))
            np.savez(original_features, h=h, rms=rms)
        for name, model in models.items():
            output = output_dir / f"{item['language']}_{name}.wav"
            key = f"{item['language']}_{name}"
            row = next((r for r in report["clips"] if r["key"] == key), None)
            if row is None:
                y = process_audio(model, x, sr)
                sf.write(output, y, sr, subtype="FLOAT")
                row = {**item, "key": key, "variant": name, "output": str(output.relative_to(ROOT)),
                    "source_sha256": digest(source), "output_sha256": digest(output),
                    **quality_metrics(x, y), **speech_metrics(x, y, sr)}
                report["clips"].append(row)
                ledger.write_text(json.dumps(report, indent=2))
            elif digest(output) != row["output_sha256"]:
                raise ValueError("Saved feedback audio was changed")
            cache = feature_dir / f"{key}.npz"
            if not cache.exists():
                y, _ = sf.read(output)
                h, rms = frames.frames(frames.to16k(y, sr))
                np.savez(cache, h=h, rms=rms)
            row["features"] = str(cache.relative_to(ROOT))
            if source_hash in source_labels:
                row["original"] = source_labels[source_hash]
            row["original_features"] = str(original_features.relative_to(ROOT))
            ledger.write_text(json.dumps(report, indent=2))
            if args.prepare_only:
                print(f"Prepared {key}", flush=True)
                continue
            if "edited" in row:
                continue
            try:
                code, response = cartesia.detect(output)
            except cartesia.RateLimited:
                report["next_allowed_at"] = time.time()+300
                ledger.write_text(json.dumps(report, indent=2))
                print("Rate limited; saved progress and a five-minute cooldown", flush=True)
                return 2
            except Exception as error:
                print(f"Request failed ({type(error).__name__}); progress saved", flush=True)
                return 2
            if code != 200 or not isinstance(response, dict) or not isinstance(response.get("detected"), bool):
                print(f"Unexpected HTTP {code}; progress saved", flush=True)
                return 2
            row["edited"] = response["detected"]
            row["response_keys"] = sorted(response)
            row["numeric_response"] = {k:v for k,v in response.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}
            ledger.write_text(json.dumps(report, indent=2))
            print(f"{key}: Cartesia={row['edited']} PESQ={row['pesq_wb']:.3f}; fields={row['response_keys']}", flush=True)
    print(f"Completed {sum('edited' in r for r in report['clips'])} labelled edits on {len(plan)} training voices", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
