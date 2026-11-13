"""Apply transforms to audio and query the Cartesia watermark detector, with caching."""
import hashlib, json, pathlib, subprocess, sys, tempfile
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "detect_cache.json"
CACHE.parent.mkdir(exist_ok=True)
_cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
TMP = ROOT / "audio" / "tmp"; TMP.mkdir(parents=True, exist_ok=True)


def detect_array(x, sr, tag=""):
    x = np.asarray(x, dtype=np.float32)
    p = TMP / "probe.wav"
    sf.write(p, x, sr, subtype="FLOAT")
    h = hashlib.sha1(p.read_bytes()).hexdigest()
    if h not in _cache:
        code, res = c.detect(p)
        if code != 200:
            return res
        _cache[h] = {"code": code, "res": res, "tag": tag}
        CACHE.write_text(json.dumps(_cache, indent=1))
    r = _cache[h]
    return r["res"].get("detected") if isinstance(r["res"], dict) and "detected" in r["res"] else r["res"]


def load(p):
    x, sr = sf.read(p, dtype="float64")
    return (x if x.ndim == 1 else x.mean(1)), sr


def ffmpeg_roundtrip(x, sr, args, ext):
    with tempfile.TemporaryDirectory() as d:
        a, b, o = f"{d}/a.wav", f"{d}/b.{ext}", f"{d}/o.wav"
        sf.write(a, x, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, *args, b], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", b, "-ar", str(sr), "-ac", "1", "-c:a", "pcm_f32le", o], check=True)
        y, _ = load(o)
    return y


def butter(x, sr, kind, fc, order=8):
    sos = signal.butter(order, fc, btype=kind, fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x)


def resample(x, sr, new):
    from math import gcd
    g = gcd(sr, new)
    return signal.resample_poly(x, new // g, sr // g)


def noise(x, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    p = np.mean(x ** 2)
    return x + rng.standard_normal(len(x)) * np.sqrt(p / 10 ** (snr_db / 10))
