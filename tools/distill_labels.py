"""Collect Cartesia detector labels for distillation.

Stage 1: natural clips on held-out voices (24-29) -> Cartesia's real TPR / FPR per model.
Stage 2: boundary transforms of training-voice clips (voices 0-23, minus 1 and 5 which appear in the
         validation set). These are the confound-breaking labels: same timbre, different mark state.
Audio for every labelled item is saved so training can use it directly.
"""
import json, pathlib, random, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from probe import *

OUT = ROOT / "audio" / "distill"; OUT.mkdir(parents=True, exist_ok=True)
LEDGER = ROOT / "results" / "distill.json"
done = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
MODELS = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
TWIN = {"sonic-3.6": "sonic-3.5", "sonic-preview": "sonic-3", "sonic-3.5": "sonic-3.6", "sonic-3": "sonic-preview"}


def label(name, x, sr, meta):
    if name in done:
        return
    p = OUT / f"{name}.wav"; sf.write(p, np.asarray(x, np.float32), sr, subtype="FLOAT")
    for _ in range(12):
        try:
            res = detect_array(x, sr, name); break
        except c.RateLimited:
            print(f"  429 on {name}, cooling 5 min", flush=True); time.sleep(300)
        except c.EgressChanged as e:
            raise SystemExit(str(e))
    if not isinstance(res, bool):
        print(f"  skip {name}: {res}", flush=True); return
    done[name] = dict(meta, detected=res, path=str(p.relative_to(ROOT)))
    LEDGER.write_text(json.dumps(done, indent=1))
    print(f"{name:48s} {res}", flush=True)


def clip(t, v, m):
    return load(ROOT / f"audio/corpus3/t{t}_v{v:02d}_{m}.wav")


# Stage 2: boundary transforms on training voices
rng = random.Random(0)
train_voices = [v for v in range(24) if v not in (1, 5)]


def bandshift(x, sr, ms, lo=1500, hi=3500):
    b = butter(x, sr, "band", [lo, hi]); d = int(round(ms * sr / 1000))
    return x - b + np.concatenate([np.zeros(d), b[:len(b) - d]])


FAMILIES = {
    "snr": lambda x, y, sr, r: (noise(x, r.uniform(12, 18), seed=r.randrange(1 << 30)), "snr"),
    "splice": lambda x, y, sr, r: (lambda k: (np.concatenate([x[:k], y[k:min(len(y), len(x))]]), f"splice{k / sr:.1f}s"))(int(r.uniform(0.8, 2.5) * sr)),
    "bandshift": lambda x, y, sr, r: (lambda ms: (bandshift(x, sr, ms), f"shift{ms:.0f}ms"))(r.uniform(20, 120)),
    "lowpass": lambda x, y, sr, r: (lambda fc: (butter(x, sr, "low", fc), f"lp{fc:.0f}"))(r.uniform(1800, 4200)),
    "highpass": lambda x, y, sr, r: (lambda fc: (butter(x, sr, "high", fc), f"hp{fc:.0f}"))(r.uniform(1000, 3500)),
    "mix": lambda x, y, sr, r: (lambda a: (a * x[:min(len(x), len(y))] + (1 - a) * y[:min(len(x), len(y))], f"mix{a:.2f}"))(r.uniform(0.2, 0.8)),
    "bandswap": lambda x, y, sr, r: (lambda n: (x[:n] - butter(x[:n], sr, "band", [1500, 3500]) + butter(y[:n], sr, "band", [1500, 3500]), "bandswap"))(min(len(x), len(y))),
}
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 200):
    fam = rng.choice(list(FAMILIES)); m = rng.choice(MODELS); v = rng.choice(train_voices); t = rng.randrange(6)
    x, sr = clip(t, v, m); y, _ = clip(t, v, TWIN[m])          # twin = same text & voice, other mark state
    r = random.Random(i)
    z, desc = FAMILIES[fam](x, y, sr, r)
    if len(z) < 5 * sr:
        z = np.concatenate([z, np.zeros(5 * sr - len(z) + 1)])
    label(f"tf{i:03d}_{fam}_{desc}_t{t}_v{v:02d}_{m}", z, sr, dict(stage="transform", family=fam, desc=desc, model=m, voice=v, text=t))

# Stage 1 (run after the transforms): natural clips, held-out voices
for v in (24, 27):          # two held-out voices: Cartesia TPR / FPR estimate
    for t in range(6):
        for m in MODELS:
            x, sr = clip(t, v, m)
            label(f"nat_t{t}_v{v:02d}_{m}", x, sr, dict(stage="natural", model=m, voice=v, text=t))
