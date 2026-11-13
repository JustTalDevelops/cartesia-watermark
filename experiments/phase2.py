"""Phase-2 probes against Cartesia's detector (formats, filters, splices, noise), resumable ledger results/phase2.json.
"""
import sys, pathlib, glob
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
from probe import *

import json, os, time
calls = [0]
LEDGER = ROOT / "results" / "phase2.json"
done = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def say(*a): print(*a, flush=True)


def run(name, fn):
    """Run one detector experiment once; resume-safe; waits out 429s on the same IP."""
    if name not in done:
        for attempt in range(6):
            try:
                done[name] = fn()
                calls[0] += 1
                break
            except c.RateLimited:
                if os.environ.get("STOP_ON_429"):
                    raise SystemExit(f"429 on {name} after {calls[0]} calls this run")
                say(f"  429 on {name}, cooling down 10 min ({attempt + 1}/6)")
                time.sleep(600)
        else:
            raise SystemExit("still rate-limited after 1 h of cooldowns; stopping")
        LEDGER.write_text(json.dumps(done, indent=1))
    say(f"{name:45s} {done[name]}")
P0, sr = load("audio/corpus/t0_daniel_sonic-3.6.wav"); N0, _ = load("audio/corpus/t0_daniel_sonic-3.5.wav")

say("== A. repeatability (uncached)")
p = TMP / "rep.wav"
for name, x in [("neg_orig", N0), ("neg_lp8k", butter(N0, sr, "low", 8000)), ("neg_last6s", N0[-6*sr:]), ("pos_snr10", noise(P0, 10))]:
    sf.write(p, x.astype(np.float32), sr, subtype="FLOAT")
    def twice(x=x):
        sf.write(p, x.astype(np.float32), sr, subtype="FLOAT")
        return [c.detect(p)[1].get("detected") for _ in range(2)]
    run(f"A {name}", twice)

say("== B. API output formats (sonic-3.6 unless noted)")
for f in sorted(glob.glob("audio/fmt/*")):
    run(f"B {f}", lambda f=f: c.detect(f)[1].get("detected"))

say("== C. negatives")
for f in sorted(glob.glob("audio/corpus/*sonic-3.5.wav"))[1:]:
    run(f, lambda: detect_array(load(f)[0], sr, "neg"))
rng = np.random.default_rng(1)
for name, x in [("zeros6s", np.zeros(6*sr)), ("white", 0.05*rng.standard_normal(8*sr)),
                ("sine1k", 0.2*np.sin(2*np.pi*1000*np.arange(8*sr)/sr)),
                ("lpnoise", butter(0.1*rng.standard_normal(8*sr), sr, "low", 3000, 2))]:
    run(name, lambda: detect_array(x, sr, name))

say("== D. band transplant")
def band(x, lo, hi): return butter(x, sr, "band", [lo, hi])
n = min(len(P0), len(N0)); P, N = P0[:n], N0[:n]
for lo, hi in [(2000, 4000), (1000, 4000), (1500, 3500)]:
    run(f"3.6 only {lo}-{hi}", lambda: detect_array(band(P, lo, hi), sr, "bp"))
    run(f"3.5 + 3.6[{lo}-{hi}] swapped in", lambda: detect_array(N - band(N, lo, hi) + band(P, lo, hi), sr, "tx"))
    run(f"3.6 with 3.5[{lo}-{hi}] swapped in", lambda: detect_array(P - band(P, lo, hi) + band(N, lo, hi), sr, "tx"))

say("== E. splices / sliding windows")
for frac in [0.75, 0.5, 0.25]:
    k = int(frac*n)
    run(f"3.6 first {frac:.0%} then 3.5", lambda: detect_array(np.concatenate([P[:k], N[k:]]), sr, "splice"))
L = len(P0)/sr
for s0 in np.arange(0, L-5, 2.5):
    run(f"3.6 window {s0:.1f}-{s0+5:.1f}s", lambda: detect_array(P0[int(s0*sr):int((s0+5)*sr)], sr, "win"))
