"""Cartesia detector labels for every sealed2 clip (final comparison only; results are not printed)."""
import json, pathlib, sys, time
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
import cartesia as c
import numpy as np, soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/sealed2_cartesia.json"
done = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
for p in sorted((ROOT / "audio/sealed2").glob("*.wav")):
    if p.name in done:
        continue
    x, sr = sf.read(p)
    q = p
    if len(x) < 5.2 * sr:                             # detector minimum is 5 s: pad with trailing digital silence
        q = ROOT / "audio/tmp" / f"pad_{p.name}"; q.parent.mkdir(exist_ok=True)
        sf.write(q, np.concatenate([x, np.zeros(int(5.2 * sr) - len(x))]).astype(np.float32), sr, subtype="FLOAT")
    for _ in range(12):
        try:
            code, res = c.detect(q); break
        except c.RateLimited:
            print("429, cooling 5 min", flush=True); time.sleep(300)
        except c.EgressChanged as e:
            raise SystemExit(str(e))
    if code == 200 and isinstance(res, dict) and "detected" in res:
        done[p.name] = res["detected"]; LEDGER.write_text(json.dumps(done))
    else:
        print("unexpected", code, str(res)[:120], flush=True)
    print(f"{len(done)} labelled", flush=True)
print("finished", flush=True)
