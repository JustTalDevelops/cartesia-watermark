"""Raw-sample inspection of TTS output: quantisation grid, leading/trailing silence, spectra.
"""
import sys, numpy as np, soundfile as sf
from scipy import signal
for f in sys.argv[1:]:
    x, sr = sf.read(f)
    # quantisation grid / unique values
    nz = x[x != 0]
    mant = np.abs(nz) / 2.0 ** np.floor(np.log2(np.abs(nz)))
    bits = max(len(np.binary_repr(int(round((m - 1) * 2**23)))) for m in mant[:5000])
    # leading/trailing silence stats
    a = np.abs(x); thr = 1e-3
    on = np.argmax(a > thr); off = len(x) - np.argmax(a[::-1] > thr)
    lead, tail = x[:on], x[off:]
    def db(v): return 20*np.log10(np.sqrt(np.mean(v**2))+1e-20) if len(v) else float('nan')
    f_, P = signal.welch(x, sr, nperseg=4096)
    bands = [(0,1e3),(1e3,4e3),(4e3,8e3),(8e3,12e3),(12e3,16e3),(16e3,20e3),(20e3,22.05e3)]
    bp = [10*np.log10(P[(f_>=lo)&(f_<hi)].mean()+1e-30) for lo,hi in bands]
    print(f"{f:32s} zeros={np.mean(x==0):.3f} lead={on/sr:.2f}s({db(lead):.0f}dB) tail={(len(x)-off)/sr:.2f}s({db(tail):.0f}dB) bands=" + " ".join(f"{b:6.1f}" for b in bp))
