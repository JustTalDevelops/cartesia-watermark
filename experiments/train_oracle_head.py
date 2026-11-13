"""Repair the surrogate with real labels on generated training-voice edits.

Replay natural training audio to retain the original task. The four earlier
development voices are held out from this repair and measure API agreement.
sealed4 is never read.
"""
import os
os.environ["FEAT"] = "xlsr2"
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

import frames as fr
import train_head as th
from remover import ROOT


def windows(path, cache):
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        x, sr = sf.read(path)
        h, rms = fr.frames(fr.to16k(x if x.ndim == 1 else x.mean(1), sr))
        np.savez(cache, h=h, rms=rms)
    data = np.load(cache)
    return th.windows_of(data["h"], data["rms"])


@torch.no_grad()
def evaluate(head, items):
    head.eval()
    scores, labels = [], []
    for item in items:
        score = float(head(torch.tensor(np.array(item["windows"])).float()).mean())
        scores.append(score); labels.append(item["label"])
    scores, labels = np.array(scores), np.array(labels)
    bce = float(np.mean(np.logaddexp(0, scores) - labels * scores))
    return {"clips": len(labels), "bce": bce, "accuracy": float(np.mean((scores>0)==labels)),
            "auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--calibration-development", action="store_true",
                        help="Use the earlier four development voices for oracle calibration; validate on fresh feedback edits")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--minimum-edits", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(2031)
    rng = random.Random(2031)
    manifest = json.loads((ROOT / "results/remover_wave2c.manifest.json").read_text())
    train_voices = {r["voice"] for r in manifest["train"]}
    val_voices = {r["voice"] for r in manifest["validation"]}
    feature_root = ROOT / "features_oracle/head_repair"
    X, Y, W = [], [], []
    rows = manifest["train"].copy(); rng.shuffle(rows)
    for row in rows[:512]:
        path = ROOT / row["path"]
        cache = ROOT / "features_xlsr2/clean" / path.parent.name / f"{path.stem}.npz"
        if not cache.exists():
            cache = feature_root / "natural" / f"{path.parent.name}_{path.stem}.npz"
        wins = windows(path, cache)
        if wins:
            X.append(rng.choice(wins)); Y.append(row["marked"]); W.append(1.0)
    natural = []
    for row in manifest["validation"]:
        path = ROOT / row["path"]
        cache = ROOT / "features_xlsr2/clean" / path.parent.name / f"{path.stem}.npz"
        if not cache.exists():
            cache = feature_root / "natural" / f"{path.parent.name}_{path.stem}.npz"
        wins = windows(path, cache)
        if wins:
            natural.append({"path": row["path"], "label": row["marked"], "windows": wins})
    api_validation, used = [], set()
    source_names = {"en_118_sonic-preview.wav", "es_228_sonic-preview.wav", "ja_335_sonic-3.6.wav", "pt_346_sonic-3.6.wav"}
    for ledger in sorted((ROOT / "results").glob("remover_*_cartesia.json")):
        if "oracle_round" in ledger.name:
            continue
        data = json.loads(ledger.read_text())
        for row in data.get("clips", []):
            if "edited" not in row or Path(row["source"]).name not in source_names:
                continue
            path = ROOT / row["output"]
            if not path.exists():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["output_sha256"] or digest in used:
                continue
            wins = windows(path, feature_root / "api_validation" / f"{digest}.npz")
            if wins:
                api_validation.append({"path": str(path), "label": row["edited"], "windows": wins})
                used.add(digest)
    head = th.Conv()
    head.load_state_dict(torch.load(ROOT / "results/head_conv_heldout_x4c0.pt", weights_only=True))
    calibration = []
    feedback = json.loads((ROOT / "results/remover_oracle_round1_cartesia.json").read_text())
    if args.calibration_development:
        calibration, api_validation = api_validation, []
        originals = set()
        for row in feedback["clips"]:
            if "edited" in row:
                data = np.load(ROOT / row["features"])
                wins = th.windows_of(data["h"], data["rms"])
                if wins:
                    api_validation.append({"path": row["output"], "label": row["edited"], "windows": wins})
            if "original" in row and row["source_sha256"] not in originals:
                data = np.load(ROOT / row["original_features"])
                wins = th.windows_of(data["h"], data["rms"])
                if wins:
                    api_validation.append({"path": row["source"], "label": row["original"], "windows": wins})
                    originals.add(row["source_sha256"])
        print(f"Oracle calibration: {len(calibration)} earlier edits; new API validation: {len(api_validation)} edits/baselines", flush=True)
    before = {"api": evaluate(head, api_validation), "natural": evaluate(head, natural)}
    print("Baseline", json.dumps(before), flush=True)
    if args.prepare_only:
        print("Replay and validation features prepared", flush=True)
        return
    count, originals = 0, set()
    for item in calibration:
        for win in item["windows"]:
            X.append(win);Y.append(item["label"]);W.append(15.0)
        count += 1
    for row in ([] if args.calibration_development else feedback["clips"]):
        if row["voice"] not in train_voices or row["voice"] in val_voices:
            raise ValueError("Feedback voice is outside the training partition")
        if "edited" not in row:
            continue
        data = np.load(ROOT / row["features"])
        for win in th.windows_of(data["h"], data["rms"]):
            X.append(win);Y.append(row["edited"]);W.append(15.0)
        count += 1
        if "original" in row and row["source_sha256"] not in originals:
            data = np.load(ROOT / row["original_features"])
            for win in th.windows_of(data["h"], data["rms"]):
                X.append(win);Y.append(row["original"]);W.append(5.0)
            originals.add(row["source_sha256"])
    if count < args.minimum_edits:
        raise SystemExit(f"Need at least {args.minimum_edits} labelled training edits; currently {count}")
    X = torch.tensor(np.array(X, dtype=np.float16));Y = torch.tensor(Y,dtype=torch.float32);W = torch.tensor(W)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        head.train(); order=torch.randperm(len(X)); losses=[]
        for start in range(0,len(X),32):
            batch=order[start:start+32]
            loss=(F.binary_cross_entropy_with_logits(head(X[batch].float()),Y[batch],reduction="none")*W[batch]).mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        metrics={"epoch":epoch+1,"loss":float(np.mean(losses)),"api":evaluate(head,api_validation),"natural":evaluate(head,natural)}
        if calibration:
            metrics["calibration"] = evaluate(head, calibration)
        history.append(metrics);print(json.dumps(metrics),flush=True)
        selection=metrics["api"]["bce"]+0.2*metrics["natural"]["bce"]
        if calibration:
            selection += 0.5 * metrics["calibration"]["bce"]
        if selection<best:
            best=selection
            torch.save(head.state_dict(),ROOT/f"results/head_conv_heldout_oracle{args.round}.pt")
    calibration_voices = []
    if calibration:
        plan = json.loads((ROOT / "audio/corpus5_plan.json").read_text())
        calibration_voices = sorted({plan[int(name.split('_')[1])][2] for name in source_names})
    (ROOT/f"results/oracle_detector_r{args.round}.json").write_text(json.dumps({"members":[f"oracle{args.round}"],"feat":"xlsr2","threshold":0.0,"calibration_voices":calibration_voices},indent=2))
    (ROOT/f"results/oracle_detector_r{args.round}_report.json").write_text(json.dumps({"feedback_edits":count,"calibration_voices":calibration_voices,"source_baselines":len(originals),"replay_windows":512,"before":before,"history":history},indent=2))


if __name__ == "__main__":
    main()
