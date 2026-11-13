"""Measure a neural reconstruction prior and band-limited reconstruction blends."""
import os
os.environ["FEAT"] = "xlsr2"
import hashlib
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import numpy as np
from scipy import signal
import soundfile as sf
import torch
from transformers import DacModel
from huggingface_hub import hf_hub_download

from remover import ROOT, quality_metrics, resample
from eval_remover import speech_metrics
from final_detector import Detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=("dac", "vocos"), default="dac")
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.codec == "dac":
        config_file = Path(hf_hub_download("descript/dac_44khz", "config.json"))
        revision = config_file.parent.name
        model = DacModel.from_pretrained("descript/dac_44khz", revision=revision).eval().requires_grad_(False)
        codec_rate, hop = 44100, 512
    else:
        from third_party.vocos_mel import VocosMel, REVISION
        revision = REVISION
        model = VocosMel.pretrained().eval().requires_grad_(False)
        codec_rate, hop = 24000, 256
    detector = Detector(config=ROOT / "results/final_detector_r4.json")
    sources = json.loads((ROOT / "results/remover_wave1_probe250_cartesia.json").read_text())["clips"]
    out = ROOT / "audio" / f"{args.codec}_probe"
    out.mkdir(exist_ok=True)
    rows = []
    started = time.time()
    for source in sources:
        path = ROOT / source["source"]
        x, sr = sf.read(path)
        native = resample(x, sr, codec_rate)
        padded = np.pad(native, (0, (-len(native)) % hop))
        with torch.inference_mode():
            tensor = torch.tensor(padded, dtype=torch.float32)[None]
            decoded = (model(tensor[:, None]).audio_values if args.codec == "dac" else model(tensor)).numpy().reshape(-1)
        decoded = resample(decoded[:len(native)], codec_rate, sr)[:len(x)]
        if len(decoded) < len(x):
            decoded = np.pad(decoded, (0, len(x) - len(decoded)))
        corr = signal.correlate(x, decoded, mode="full", method="fft")
        center = len(decoded)-1
        span = int(sr * 0.02)
        lag = int(np.argmax(corr[center-span:center+span+1]) - span)
        if lag > 0:
            decoded = np.pad(decoded, (lag, 0))[:len(x)]
        elif lag < 0:
            decoded = np.pad(decoded[-lag:], (0, -lag))
        variants = {"full": decoded}
        for low, high in ((1500,3500), (1000,4000), (500,5000)):
            sos = signal.butter(6, [low, high], btype="bandpass", fs=sr, output="sos")
            change = signal.sosfiltfilt(sos, decoded-x)
            for strength in (0.5, 0.75, 1.0):
                variants[f"band{low}_{high}_s{strength}"] = x + strength*change
        for name, y in variants.items():
            output = out / f"{path.stem}_{name}.wav"
            sf.write(output,y,sr,subtype="FLOAT")
            row = {"source": str(path.relative_to(ROOT)), "output": str(output.relative_to(ROOT)),
                "variant": name, "lag_samples":lag, "score":detector.score(y,sr),
                **quality_metrics(x,y), **speech_metrics(x,y,sr)}
            rows.append(row)
            print(path.name,name,json.dumps({k:row[k] for k in ('score','snr_db','pesq_wb','estoi')}),flush=True)
        (ROOT / "results" / f"{args.codec}_probe.json").write_text(json.dumps({"codec":args.codec,
            "codec_revision":revision,"threshold":detector.threshold,"clips":rows},indent=2))
    print(f"Finished in {time.time()-started:.0f}s",flush=True)


if __name__ == "__main__":
    main()
