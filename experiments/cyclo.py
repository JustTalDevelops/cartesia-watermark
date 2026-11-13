"""Cyclostationarity test: does the 2-3.5 kHz band carry a fixed symbol/modulation rate?
Square the band signal, divide out the slow speech envelope, and look for spectral lines shared by
both marked models but absent from both unmarked ones."""
import glob, pathlib
import numpy as np, soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
ROOT = pathlib.Path(__file__).resolve().parent.parent
SR = 44100
sos = signal.butter(6, [2000, 3500], btype="band", fs=SR, output="sos")
NF = 2 ** 18                                           # ~6 s window -> 0.17 Hz resolution
models = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]


def cyc(p):
    x, sr = sf.read(p)
    e = signal.sosfiltfilt(sos, x) ** 2
    slow = uniform_filter1d(e, int(0.05 * SR)) + 1e-12  # 50 ms speech envelope
    r = e / slow - 1
    r = r[:NF] if len(r) >= NF else np.pad(r, (0, NF - len(r)))
    return np.abs(np.fft.rfft(r * np.hanning(NF))) ** 2


f = np.fft.rfftfreq(NF, 1 / SR)
keep = (f > 5) & (f < 3000)
S = {}
for m in models:
    S[m] = np.mean([cyc(p) for p in sorted(glob.glob(str(ROOT / f"audio/corpus3/*_{m}.wav")))], 0)[keep]
fk = f[keep]
for m in models:                                       # whiten each model's spectrum by its own smooth trend
    S[m] = S[m] / uniform_filter1d(S[m], 301)
marked = np.minimum(S["sonic-3.6"], S["sonic-preview"]); unmarked = np.maximum(S["sonic-3.5"], S["sonic-3"])
ratio = marked / unmarked
top = np.argsort(-ratio)[:15]
print("candidate lines (Hz): marked-min / unmarked-max peak ratio, and each model's whitened level")
for i in sorted(top, key=lambda i: fk[i]):
    print(f"  {fk[i]:8.2f} Hz  ratio={ratio[i]:5.2f}   " + "  ".join(f"{m}={S[m][i]:5.2f}" for m in models))
print("\nstrongest whitened peaks per model (Hz):")
for m in models:
    pk = np.argsort(-S[m])[:6]
    print(f"  {m:14s} " + ", ".join(f"{fk[i]:.1f}({S[m][i]:.1f})" for i in sorted(pk, key=lambda i: fk[i])))
