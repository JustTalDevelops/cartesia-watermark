"""Frame-level SSL features for every clip and augmentation variant (resumable, sharded).

    [FEAT=xlsr2] python tools/frames.py <shard> <nshards> [threads]

FEAT=wavlm1 (default): WavLM-base-plus layer 1 (768-d) -> features/
FEAT=xlsr2           : XLS-R 300M layer 2 (1024-d)      -> features_xlsr2/
Writes <root>/<variant>/<corpus>/<stem>.npz with h (T x D, float16) and rms (T, per 20 ms frame).
Variants: clean, noisy (17-30 dB SNR), mp3 (32/64 kbps), reversed (marked clips only).
"""
import glob, pathlib, subprocess, sys, tempfile
import numpy as np, soundfile as sf, torch
from scipy import signal
from transformers import WavLMModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Feature set, chosen by env FEAT: "wavlm1" (default; WavLM-base-plus layer 1, 768-d, features/) or
# "xlsr2" (XLS-R 300M layer 2, 1024-d, features_xlsr2/; multilingual, better on non-English clips).
FEAT = __import__("os").environ.get("FEAT", "wavlm1")
OUT = ROOT / ("features" if FEAT == "wavlm1" else f"features_{FEAT}")
LAYER = {"wavlm1": 1, "xlsr2": 2}[FEAT]
MARKED = ("sonic-3.6", "sonic-preview")
_m = None


def model():
    global _m
    if _m is None:
        if FEAT == "wavlm1":
            _m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
            _m.encoder.layers = _m.encoder.layers[:1]       # only layer 1 is needed (verified identical)
        else:
            from transformers import Wav2Vec2Model
            _m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").eval()
            _m.encoder.layers = _m.encoder.layers[:3]       # keep one layer past 2: the last output gets the final LayerNorm
    return _m


def to16k(x, sr):
    from math import gcd
    g = gcd(int(sr), 16000)
    return signal.resample_poly(np.asarray(x, np.float64), 16000 // g, int(sr) // g)


def frames(y16):
    """16 kHz audio -> (layer-1 frames float16, per-frame RMS)."""
    if len(y16) < 16000:
        y16 = np.pad(y16, (0, 16000 - len(y16)))
    yn = ((y16 - y16.mean()) / (y16.std() + 1e-9)).astype(np.float32)
    with torch.no_grad():
        h = model()(torch.tensor(yn)[None], output_hidden_states=True).hidden_states[LAYER][0].numpy()
    T = len(h)
    rms = np.array([np.sqrt(np.mean(y16[i * 320:(i + 1) * 320] ** 2)) if i * 320 < len(y16) else 0.0 for i in range(T)])
    return h.astype(np.float16), rms.astype(np.float32)


def mp3(x, sr, kbps):
    with tempfile.TemporaryDirectory() as d:
        sf.write(f"{d}/a.wav", x, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/a.wav", "-b:a", f"{kbps}k", f"{d}/b.mp3"], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/b.mp3", "-ar", str(sr), "-ac", "1", f"{d}/c.wav"], check=True)
        y, _ = sf.read(f"{d}/c.wav")
    return y


def clip_list():
    out = []
    for corpus in ("corpus3", "corpus4", "corpus5", "corpus6", "corpus7", "sealed", "sealed2", "sealed3"):  # spent sealed sets -> training
        out += sorted(glob.glob(str(ROOT / f"audio/{corpus}/*.wav")))
    return out


if __name__ == "__main__":
    shard, nshards = int(sys.argv[1]), int(sys.argv[2])
    torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    paths = clip_list()[shard::nshards]
    for i, p in enumerate(paths):
        p = pathlib.Path(p); corpus = p.parent.name; stem = p.stem
        marked = stem.rsplit("_", 1)[-1] in MARKED or any(stem.endswith(m) for m in MARKED)
        rng = np.random.default_rng(abs(hash(stem)) % (1 << 32))
        x = None
        for variant in ["clean", "noisy", "mp3"] + (["reversed"] if marked else []):
            out = OUT / variant / corpus / f"{stem}.npz"
            if out.exists():
                continue
            if x is None:
                x, sr = sf.read(p)
            if variant == "clean":
                z = x
            elif variant == "noisy":
                snr = rng.uniform(17, 30); z = x + rng.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))
            elif variant == "mp3":
                z = mp3(x, sr, int(rng.choice([32, 64])))
            else:
                z = x[::-1].copy()
            h, rms = frames(to16k(z, sr))
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, h=h, rms=rms)
        if i % 100 == 0:
            print(f"shard {shard}: {i}/{len(paths)}", flush=True)
    print(f"shard {shard} done", flush=True)
