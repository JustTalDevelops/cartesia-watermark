"""Window-level WavLM layer-1 features for the windowed detector.

For every natural clip (corpus3 + corpus4): clean, one noisy copy (17-30 dB SNR, label kept), and for marked
clips a reversed copy (label 0). Plus synthetic noise negatives. Windows: 3 s (150 frames), hop 1.5 s,
skipped when quieter than -50 dBFS. Features: mean+std of layer-1 frames within the window.
"""
import glob, pathlib, sys
import numpy as np, soundfile as sf, torch
from scipy import signal
from transformers import WavLMModel
ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER, WIN, HOP, GATE = 1, 150, 75, 10 ** (-50 / 20)
BAND = "--band" in sys.argv or bool(__import__("os").environ.get("WAVLM_BAND"))   # feed only 1-4.5 kHz
_sos = signal.butter(6, [1000, 4500], btype="band", fs=16000, output="sos")
_m = None


def model():
    global _m
    if _m is None:
        torch.set_num_threads(20); _m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
    return _m


def to16k(x, sr):
    from math import gcd
    g = gcd(int(sr), 16000)
    return signal.resample_poly(np.asarray(x, np.float64), 16000 // g, int(sr) // g)


def windows(x, sr):
    """Audio -> list of window features (empty if no window passes the loudness gate)."""
    y = to16k(x, sr)
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))
    yb = signal.sosfiltfilt(_sos, y) if BAND else y
    yn = ((yb - yb.mean()) / (yb.std() + 1e-9)).astype(np.float32)
    with torch.no_grad():
        h = model()(torch.tensor(yn)[None], output_hidden_states=True).hidden_states[LAYER][0].numpy()
    out = []
    starts = list(range(0, max(1, len(h) - WIN + 1), HOP))
    for s in starts:
        seg = y[s * 320:(s + WIN) * 320]
        if np.sqrt(np.mean(seg ** 2)) < GATE:
            continue
        w = h[s:s + WIN]; out.append(np.concatenate([w.mean(0), w.std(0)]))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []  # (clip_key, variant, label, feature)
    paths = sorted(glob.glob(str(ROOT / "audio/corpus3/*.wav"))) + sorted(glob.glob(str(ROOT / "audio/corpus4/*.wav")))
    for i, p in enumerate(paths):
        key = str(pathlib.Path(p).relative_to(ROOT)); m = pathlib.Path(p).stem.split("_", 2)[2]
        lab = int(m in ("sonic-3.6", "sonic-preview"))
        x, sr = sf.read(p)
        for f in windows(x, sr):
            rows.append((key, "clean", lab, f))
        snr = rng.uniform(17, 30)
        xn = x + rng.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))
        for f in windows(xn, sr):
            rows.append((key, "noisy", lab, f))
        if lab:
            for f in windows(x[::-1].copy(), sr):
                rows.append((key, "reversed", 0, f))
        if i % 200 == 0:
            print(i, "/", len(paths), flush=True)
    for j in range(90):
        kind = j % 3; n = 6 * 16000
        z = rng.standard_normal(n)
        if kind == 1:
            z = np.cumsum(z); z -= np.convolve(z, np.ones(200) / 200, "same")
        elif kind == 2:
            z = signal.sosfilt(signal.butter(4, 3000, fs=16000, output="sos"), z)
        z *= 0.05 / z.std()
        for f in windows(z, 16000):
            rows.append((f"synthetic/noise{j}", "noise", 0, f))
    np.savez(ROOT / f"results/wavlm_windows{'_band' if BAND else ''}.npz", keys=np.array([r[0] for r in rows]), variant=np.array([r[1] for r in rows]),
             label=np.array([r[2] for r in rows]), feats=np.array([r[3] for r in rows], dtype=np.float32))
    print("done", len(rows), "windows", flush=True)
