"""Build unpadded, multilingual validation audio for independent API checks.

Joins both marked renders of one held-out voice/text per language. These are
validation voices; sealed4 stays untouched for the eventual final test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from remover import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/remover_wave2c.manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    excluded = {r["voice"] for r in manifest["train"]} | set(manifest.get("initialization_train_voices", []))
    reserved = {r[2] for r in json.loads((ROOT / "audio/corpus7_plan.json").read_text()) if r[0] == "sealed4"}
    pools = {}
    for row in manifest["validation"]:
        if not row["marked"]:
            continue
        if row["voice"] in excluded | reserved:
            raise ValueError("Validation voice overlaps training or the final reserved test set")
        source = ROOT / row["path"]
        prefix = source.stem.rsplit("_", 1)[0]
        pair = [source.with_name(f"{prefix}_{model}.wav") for model in ("sonic-3.6", "sonic-preview")]
        if all(p.exists() for p in pair):
            pools.setdefault(row["language"], {})[(row["voice"], prefix)] = (row, pair)
    out = ROOT / "audio/remover_validation"
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists():
        parser.error("Validation plan already exists; keep the original fixed validation set")
    rng = random.Random(2029)
    used, items = set(), []
    for language, candidates in sorted(pools.items()):
        options = list(candidates.values())
        rng.shuffle(options)
        for row, pair in options:
            if row["voice"] in used:
                continue
            x, sr = sf.read(pair[0]); y, sr2 = sf.read(pair[1])
            if sr != sr2:
                raise ValueError("Source sample rates differ")
            joined = np.concatenate([x, np.zeros(int(0.3 * sr)), y])
            if len(joined) / sr < 5:
                continue
            target = out / f"{language}_{len(items):02d}_marked.wav"
            sf.write(target, joined, sr, subtype="FLOAT")
            items.append({"path": str(target.relative_to(ROOT)), "sha256": sha(target),
                "voice": row["voice"], "language": language, "seconds": len(joined) / sr,
                "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in pair]})
            used.add(row["voice"])
            break
    plan = {"manifest_sha256": sha(args.manifest), "purpose": "validation and model selection; not a sealed test",
            "items": items}
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"Built {len(items)} clips from {len(used)} validation voices: "
          + ", ".join(r["language"] for r in items))


if __name__ == "__main__":
    main()
