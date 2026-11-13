"""Write edited audio using a trained model; no API or detector at inference.

    .venv/bin/python tools/remove_watermark.py input.wav output.wav

Detector suppression does not certify that a watermark has been removed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import soundfile as sf
import torch

from remover import DEFAULT_CHECKPOINT, load_model, process_audio, quality_metrics


def read_audio(path):
    try:
        return sf.read(path, always_2d=False)
    except (sf.LibsndfileError, RuntimeError):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "decoded.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                            "-c:a", "pcm_f32le", str(target)], check=True)
            return sf.read(target, always_2d=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--chunk-seconds", type=float, default=6)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or (args.output.exists() and
                                                       args.input.samefile(args.output)):
        parser.error("Output must differ from the source file")
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists; use --overwrite to replace it")
    if args.output.suffix.lower() != ".wav":
        parser.error("Use a .wav output path")
    if args.threads < 1:
        parser.error("Threads must be positive")
    torch.set_num_threads(args.threads)
    model, _ = load_model(args.checkpoint)
    x, sr = read_audio(args.input)
    if not len(x):
        parser.error("Input audio is empty")
    y = process_audio(model, x, sr, strength=args.strength, chunk_seconds=args.chunk_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Float WAV avoids clipping or reducing the input's precision to 16-bit PCM.
    with args.output.open("wb" if args.overwrite else "xb") as handle:
        sf.write(handle, y, sr, format="WAV", subtype="FLOAT")
    encoded, _ = sf.read(args.output)
    print(json.dumps({"output": str(args.output), "sample_rate": sr,
                      "samples": len(y), "channels": 1 if y.ndim == 1 else y.shape[1],
                      **quality_metrics(x, encoded)}, indent=2))


if __name__ == "__main__":
    main()
