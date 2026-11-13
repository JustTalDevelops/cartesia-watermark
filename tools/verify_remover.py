"""Check original and edited files with Cartesia, without padding short audio.

    .venv/bin/python tools/verify_remover.py --tag v1 audio/h2h4/en_005_marked.wav

Calls are paced by cartesia.py. Results are resumable and bound to file hashes.
Uses an existing API key; does not generate or purchase TTS audio.
"""
import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch

import cartesia
from remover import ROOT, DEFAULT_CHECKPOINT, load_model, process_audio, quality_metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--fresh", action="store_true", help="Query even when a matching file hash has a saved label")
    parser.add_argument("--edited-files", nargs="+", type=Path, help="Compare already generated WAV files, in input order")
    args = parser.parse_args()
    if not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("Use an alphanumeric tag")
    if not cartesia._env("CARTESIA_API_KEY"):
        parser.error("CARTESIA_API_KEY is required for independent verification")
    torch.set_num_threads(2)
    if args.edited_files and len(args.edited_files) != len(args.inputs):
        parser.error("Provide one edited file for every input")
    model = None
    if not args.edited_files:
        model, _ = load_model(args.checkpoint)
    out = ROOT / "audio" / f"remover_{args.tag}" / "official"
    out.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / "results" / f"remover_{args.tag}_cartesia.json"
    checkpoint_hash = digest(args.checkpoint)
    report = json.loads(ledger.read_text()) if ledger.exists() else {
        "checkpoint_sha256": checkpoint_hash, "api_version": cartesia.VERSION, "clips": []}
    if report["checkpoint_sha256"] != checkpoint_hash:
        parser.error("Ledger belongs to a different checkpoint; use a new tag")
    cache = {}
    if not args.fresh:
        for previous in sorted((ROOT / "results").glob("remover_*_cartesia.json")):
            saved = json.loads(previous.read_text())
            if saved.get("api_version", cartesia.VERSION) != cartesia.VERSION:
                continue
            for item in saved.get("clips", []):
                for label, key in (("original", "source_sha256"), ("edited", "output_sha256")):
                    if label in item and key in item:
                        if item[key] in cache and cache[item[key]][0] != item[label]:
                            parser.error("Conflicting saved Cartesia labels for the same file hash")
                        cache[item[key]] = (item[label], previous.name)
    for index, path in enumerate(args.inputs):
        x, sr = sf.read(path)
        if len(x) / sr < 5:
            parser.error(f"{path.name} is under 5 seconds; padding would invalidate the comparison")
        source_hash = digest(path)
        row = next((r for r in report["clips"] if r["source_sha256"] == source_hash), None)
        if row is not None and args.edited_files and digest(args.edited_files[index]) != row["output_sha256"]:
            parser.error("Edited file differs from this saved comparison; use a new tag")
        if row is None:
            if args.edited_files:
                output = args.edited_files[index].resolve()
                y, output_rate = sf.read(output)
                if output_rate != sr or y.shape != x.shape:
                    parser.error("Edited audio must preserve sample rate, length and channel count")
            else:
                y = process_audio(model, x, sr)
                output = out / f"{path.stem}_{source_hash[:8]}.wav"
                sf.write(output, y, sr, subtype="FLOAT")
            try:
                recorded_output = str(output.resolve().relative_to(ROOT.resolve()))
            except ValueError:
                recorded_output = str(output.resolve())
            row = {"source": str(path), "output": recorded_output,
                "source_sha256": source_hash, "output_sha256": digest(output),
                "seconds": len(x) / sr, **quality_metrics(x, y)}
            report["clips"].append(row)
            ledger.write_text(json.dumps(report, indent=2))
        output = ROOT / row["output"]
        if digest(output) != row["output_sha256"]:
            parser.error("Edited file differs from ledger; use a new tag")
        for label, target in (("original", path), ("edited", output)):
            if label in row and not args.fresh:
                continue
            cached = cache.get(digest(target))
            if cached is not None:
                row[label], row[f"{label}_cached_from"] = cached
                ledger.write_text(json.dumps(report, indent=2))
                print(f"{path.name}: {label} detected={row[label]} (matching cached file hash)", flush=True)
                continue
            try:
                code, result = cartesia.detect(target)
            except cartesia.RateLimited:
                print("Cartesia rate-limited verification; progress is saved. Rerun later.", flush=True)
                return 2
            except Exception as error:
                # Connection errors can embed credential-bearing proxy URLs.
                print(f"Cartesia verification failed ({type(error).__name__}); progress is saved.", flush=True)
                return 2
            if code != 200 or not isinstance(result, dict) or not isinstance(result.get("detected"), bool):
                print(f"Unexpected Cartesia response (HTTP {code}); progress is saved.", flush=True)
                return 2
            row[label] = result["detected"]
            row.pop(f"{label}_cached_from", None)
            ledger.write_text(json.dumps(report, indent=2))
            print(f"{path.name}: {label} detected={row[label]}", flush=True)
    completed = [r for r in report["clips"] if "original" in r and "edited" in r]
    summary = {"pairs": len(completed), "original_detected": sum(r["original"] for r in completed),
        "edited_detected": sum(r["edited"] for r in completed),
        "original_positive_edited_negative": sum(r["original"] and not r["edited"] for r in completed)}
    report["summary"] = summary
    ledger.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
