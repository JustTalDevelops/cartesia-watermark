"""Mechanism probes against Cartesia's detector: phase vs magnitude, sub-band notches,
minimum marked duration, SNR threshold, and whether hard negatives generalise across clips."""
import sys, pathlib, json, os, time
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
from probe import *

LEDGER = ROOT / "results" / "mechanism.json"
done = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def run(name, fn):
    if name not in done:
        for attempt in range(6):
            try:
                done[name] = fn(); break
            except c.RateLimited:
                print(f"  429 on {name} (egress {c._pinned_ip[0] or 'direct'}), cooling 5 min", flush=True); time.sleep(300)
            except c.EgressChanged as e:
                LEDGER.write_text(json.dumps(done, indent=1)); raise SystemExit(str(e))
        LEDGER.write_text(json.dumps(done, indent=1))
    print(f"{name:55s} {done.get(name)}", flush=True)


def stft_edit(x, sr, lo, hi, mode, seed=0):
    f, t, Z = signal.stft(x, sr, nperseg=1024)
    b = (f >= lo) & (f < hi)
    rng = np.random.default_rng(seed)
    if mode == "randphase":
        Z[b] = np.abs(Z[b]) * np.exp(1j * rng.uniform(0, 2 * np.pi, Z[b].shape))
    elif mode == "smoothmag":                          # keep phase, blur magnitude over time
        from scipy.ndimage import uniform_filter1d
        Z[b] = uniform_filter1d(np.abs(Z[b]), 9, axis=1) * np.exp(1j * np.angle(Z[b]))
    return signal.istft(Z, sr, nperseg=1024)[1][:len(x)]


def notch(x, sr, lo, hi):
    return x - butter(x, sr, "band", [lo, hi], order=6)


clips = {"A": "audio/corpus/t0_daniel_sonic-3.6.wav", "B": "audio/corpus/t1_jacqueline_sonic-3.6.wav",
         "C": "audio/corpus/t3_jacqueline_sonic-3.6.wav"}
negs = {k: v.replace("3.6", "3.5") for k, v in clips.items()}

print("== phase vs magnitude (band 1.5-3.5 kHz, and full band)")
for k in ["A", "B"]:
    x, sr = load(clips[k])
    run(f"{k} randomise phase 1.5-3.5k", lambda: detect_array(stft_edit(x, sr, 1500, 3500, "randphase"), sr))
    run(f"{k} smooth magnitude 1.5-3.5k (phase kept)", lambda: detect_array(stft_edit(x, sr, 1500, 3500, "smoothmag"), sr))
    run(f"{k} randomise phase 0-22k", lambda: detect_array(stft_edit(x, sr, 0, 22050, "randphase"), sr))

print("== sub-band notches on clip A (which 250 Hz slices are essential)")
x, sr = load(clips["A"])
for lo in range(1000, 4500, 250):
    run(f"A notch {lo}-{lo+250}", lambda: detect_array(notch(x, sr, lo, lo + 250), sr))

print("== minimum marked duration: 3.6 for first k s, then its 3.5 twin")
n_, _ = load(negs["A"])
for k in [0.5, 1, 2]:
    m = int(k * sr)
    run(f"A first {k}s marked then 3.5", lambda: detect_array(np.concatenate([x[:m], n_[m:]]), sr))

print("== SNR threshold (white noise)")
for snr in [12, 14, 16, 18]:
    run(f"A snr {snr}", lambda: detect_array(noise(x, snr), sr))

print("== do hard negatives generalise? (clips B, C)")
for k in ["B", "C"]:
    x, sr = load(clips[k]); n_, _ = load(negs[k]); n = min(len(x), len(n_))
    run(f"{k} reversed", lambda: detect_array(x[::-1].copy(), sr))
    band = lambda y: butter(y[:n], sr, "band", [1500, 3500])
    run(f"{k} 3.6 with 3.5's 1.5-3.5k band", lambda: detect_array(x[:n] - band(x) + band(n_), sr))
    run(f"{k} only 3.6's 1.5-3.5k band", lambda: detect_array(band(x), sr))
