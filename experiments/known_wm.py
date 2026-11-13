"""Is Cartesia's watermark a published scheme? Run open detectors on the Cartesia corpus."""
import glob, pathlib, sys, warnings
import numpy as np, soundfile as sf, torch
from scipy import signal
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent


def to16k(x, sr):
    from math import gcd
    g = gcd(sr, 16000)
    return signal.resample_poly(x, 16000 // g, sr // g).astype(np.float32)


from audioseal import AudioSeal

aseal = AudioSeal.load_detector("audioseal_detector_16bits")


WAVMARK_SUBSET = {("t00", "daniel"), ("t05", "katie")}
rows = []
for p in sorted(glob.glob(str(ROOT / "audio/corpus2/*.wav"))):
    t, v, m = pathlib.Path(p).stem.split("_", 2)
    x, sr = sf.read(p); y = to16k(x, sr)
    with torch.no_grad():
        prob, msg = aseal.detect_watermark(torch.tensor(y)[None, None], 16000)
        frame = aseal(torch.tensor(y)[None, None], 16000)[0][0, 1]   # per-sample P(watermarked)
    payload = None   # WavMark: 0 hits on the clips it finished; too slow for the full corpus
    rows.append((m, float(prob), float(frame.mean()), payload is not None))
    print(f"{pathlib.Path(p).stem:32s} audioseal={float(prob):.3f} wavmark={'HIT' if payload is not None else '-'}", flush=True)

print("\nsummary (mean AudioSeal prob, WavMark hit rate):")
for m in ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]:
    r = [x for x in rows if x[0] == m]
    print(f"  {m:14s} audioseal={np.mean([x[1] for x in r]):.3f}  wavmark_hits={sum(x[3] for x in r)}/{len(r)}")
