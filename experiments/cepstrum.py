"""Echo-hiding / fixed-delay test: average real cepstrum of the 1.5-3.5 kHz band per model."""
import glob, pathlib, sys
import numpy as np, soundfile as sf
from scipy import signal
ROOT = pathlib.Path(__file__).resolve().parent.parent
sos = signal.butter(6, [1500, 3500], btype="band", fs=44100, output="sos")
N = 2048


def cep(path):
    x, sr = sf.read(path)
    y = signal.sosfiltfilt(sos, x)
    f, t, Z = signal.stft(y, sr, nperseg=N)
    P = np.abs(Z) ** 2; e = P.sum(0); keep = e > e.max() * 1e-3
    L = np.log(P[:, keep] + 1e-12)
    C = np.fft.irfft(L, axis=0)[:N // 2]               # real cepstrum per frame
    return C.mean(1), None


models = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
C = {m: np.array([cep(p)[0] for p in sorted(glob.glob(str(ROOT / f"audio/corpus3/*_{m}.wav")))[:90]]) for m in models}
q = np.arange(N // 2) / 44100 * 1000                     # quefrency in ms
lo = 20                                                    # skip low quefrencies (envelope)
marked = np.vstack([C["sonic-3.6"], C["sonic-preview"]]); unmarked = np.vstack([C["sonic-3.5"], C["sonic-3"]])
d = marked.mean(0) - unmarked.mean(0)
se = np.sqrt(marked.var(0) / len(marked) + unmarked.var(0) / len(unmarked))
t = d / (se + 1e-12)
# consistency: the same quefrency must separate BOTH marked models from BOTH unmarked ones
sep = np.minimum.reduce([C[a].mean(0) - C[b].mean(0) for a in models[:2] for b in models[2:]])
sep_neg = np.minimum.reduce([C[b].mean(0) - C[a].mean(0) for a in models[:2] for b in models[2:]])
print("top quefrencies by |t| (ms, t, every-pair margin):")
for i in sorted(np.argsort(-np.abs(t[lo:]))[:12] + lo):
    print(f"  {q[i]:6.3f} ms  t={t[i]:+6.1f}  pos-margin={sep[i]:+.4f} neg-margin={sep_neg[i]:+.4f}")
print("quefrencies where all 4 pairings agree (marked > unmarked):", [round(q[i], 3) for i in np.where(sep[lo:] > 0)[0] + lo][:20])
print("... (marked < unmarked):", [round(q[i], 3) for i in np.where(sep_neg[lo:] > 0)[0] + lo][:20])
