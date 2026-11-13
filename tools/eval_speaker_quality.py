"""Measure original/edited speaker-embedding similarity on saved evaluation WAVs.

This is a descriptive voice-preservation check, not a listening score or a
watermark detector. The model's verification threshold is dataset dependent;
this tool deliberately makes no same-speaker pass/fail claim.

Primary model card: https://huggingface.co/microsoft/wavlm-base-plus-sv
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from remover import ROOT, resample


MODEL = "microsoft/wavlm-base-plus-sv"
REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new evaluation name")
    if args.threads < 1:
        parser.error("Threads must be positive")
    torch.set_num_threads(args.threads)
    metric = {"model": MODEL, "revision": REVISION, "transformers": transformers.__version__,
              "torch": torch.__version__, "sample_rate": 16000,
              "normalization": "model feature extractor then unit-length X-vector",
              "source": "https://huggingface.co/microsoft/wavlm-base-plus-sv",
              "limitation": "Embedding similarity is not human-rated quality or a calibrated identity guarantee"}
    feature = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REVISION)
    model = WavLMForXVector.from_pretrained(MODEL, revision=REVISION).eval()
    cache = ROOT / ".cache" / "speaker_quality"
    cache.mkdir(parents=True, exist_ok=True)
    embeddings = {}

    def embedding(path, expected_hash):
        file_hash = digest(path)
        if file_hash != expected_hash:
            raise ValueError(f"Audio changed since evaluation: {path}")
        if file_hash in embeddings:
            return embeddings[file_hash]
        key = hashlib.sha256(json.dumps([metric, file_hash], sort_keys=True).encode()).hexdigest()
        saved = cache / (key + ".json")
        if saved.exists():
            value = np.asarray(json.loads(saved.read_text()), dtype=np.float64)
        else:
            audio, sample_rate = sf.read(path)
            audio = audio if audio.ndim == 1 else audio.mean(1)
            audio = resample(audio, sample_rate, 16000).astype(np.float32)
            inputs = feature(audio, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                value = model(**inputs).embeddings[0].double().numpy()
            value /= np.linalg.norm(value)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite embedding: {path}")
            saved.write_text(json.dumps(value.tolist()) + "\n")
        embeddings[file_hash] = value
        return value

    comparisons, start = [], time.monotonic()
    for report_path in args.reports:
        report = json.loads(report_path.read_text())
        rows = []
        for row in report["clips"]:
            if not row["marked"]:
                continue
            before = embedding(ROOT / row["source"], row["source_sha256"])
            after = embedding(ROOT / row["output"], row["output_sha256"])
            rows.append({"source": row["source"], "output": row["output"],
                         "source_sha256": row["source_sha256"], "output_sha256": row["output_sha256"],
                         "cosine_similarity": float(np.clip(before @ after, -1, 1))})
            print(f"{report_path.stem}: {len(rows)} clips; {time.monotonic()-start:.0f}s", flush=True)
        values = [row["cosine_similarity"] for row in rows]
        if not values:
            raise ValueError(f"No marked clips in {report_path}")
        comparisons.append({"report": str(report_path), "report_sha256": digest(report_path),
                            "checkpoint_sha256": report["checkpoint_sha256"],
                            "marked_clips": len(rows), "median_cosine_similarity": float(np.median(values)),
                            "minimum_cosine_similarity": min(values), "clips": rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump({"metric": metric, "comparisons": comparisons}, handle, indent=2)
        handle.write("\n")
    print(json.dumps([{k: v for k, v in c.items() if k != "clips"} for c in comparisons], indent=2))


if __name__ == "__main__":
    main()
