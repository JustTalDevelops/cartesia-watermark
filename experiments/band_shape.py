"""Does a shared in-band spectral shape separate marked from unmarked models, across every pairing?"""
import glob, pathlib
import numpy as np, soundfile as sf
from scipy import signal
ROOT = pathlib.Path(__file__).resolve().parent.parent
models = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
edges = np.arange(500, 6001, 250)


def prof(p):
    x, sr = sf.read(p); f, P = signal.welch(x, sr, nperseg=4096)
    ref = P[(f >= 300) & (f < 1000)].mean()                     # normalise to low-band level
    return np.array([10 * np.log10(P[(f >= a) & (f < b)].mean() / ref) for a, b in zip(edges[:-1], edges[1:])])


S = {m: np.array([prof(p) for p in sorted(glob.glob(str(ROOT / f"audio/corpus3/*_{m}.wav")))]) for m in models}
print("band (Hz)     " + "  ".join(f"{m:>13s}" for m in models) + "   min-margin  voices(3.6>3.5)")
for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
    mu = [S[m][:, i].mean() for m in models]
    margin = min(mu[p] - mu[n] for p in (0, 1) for n in (2, 3))
    # per-voice consistency: fraction of (text,voice) pairs where 3.6 exceeds its 3.5 twin
    frac = np.mean(S["sonic-3.6"][:, i] > S["sonic-3.5"][:, i])
    flag = " <== all marked > all unmarked" if margin > 0.5 else ""
    print(f"{a:5d}-{b:5d}  " + "  ".join(f"{v:13.1f}" for v in mu) + f"   {margin:+6.2f}     {frac:.2f}{flag}")
