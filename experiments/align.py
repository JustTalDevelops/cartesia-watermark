"""Is the mark tied to the host at a precise alignment? Delay only the 1.5-3.5 kHz band of marked audio."""
import sys, pathlib, json, time
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
from probe import *

LEDGER = ROOT / "results" / "align.json"
done = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def run(name, fn):
    if name not in done:
        for _ in range(6):
            try:
                done[name] = fn(); break
            except c.RateLimited:
                print(f"  429 on {name}, cooling 5 min", flush=True); time.sleep(300)
            except c.EgressChanged as e:
                LEDGER.write_text(json.dumps(done, indent=1)); raise SystemExit(str(e))
        LEDGER.write_text(json.dumps(done, indent=1))
    print(f"{name:50s} {done.get(name)}", flush=True)


def shift_band(x, sr, ms, lo=1500, hi=3500):
    b = butter(x, sr, "band", [lo, hi]); d = int(round(ms * sr / 1000))
    return x - b + np.concatenate([np.zeros(d), b[:len(b) - d]])


A, sr = load("audio/corpus/t0_daniel_sonic-3.6.wav"); B, _ = load("audio/corpus/t1_jacqueline_sonic-3.6.wav")
for ms in [1, 5, 20, 100]:
    run(f"A band 1.5-3.5k delayed {ms} ms", lambda: detect_array(shift_band(A, sr, ms), sr))
n = min(len(A), len(B)); band = lambda y: butter(y[:n], sr, "band", [1500, 3500])
run("A with B's (marked) 1.5-3.5k band", lambda: detect_array(A[:n] - band(A) + band(B), sr))
