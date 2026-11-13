"""Cartesia detector labels for a list of clips, in the given order (resumable ledger).

    python tools/label_files.py results/<ledger>.json audio/a.wav audio/b.wav ...

Clips shorter than 5.2 s are padded with trailing digital silence (the detector's minimum is 5 s).
"""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c
import numpy as np, soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / sys.argv[1]
done = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
for p in map(pathlib.Path, sys.argv[2:]):
    key = f"{p.parent.name}/{p.name}"
    if key in done:
        continue
    x, sr = sf.read(p)
    q = p
    if len(x) < 5.2 * sr:
        q = ROOT / "audio/tmp" / f"pad_{p.parent.name}_{p.name}"; q.parent.mkdir(exist_ok=True)
        sf.write(q, np.concatenate([x, np.zeros(int(5.2 * sr) - len(x))]).astype(np.float32), sr, subtype="FLOAT")
    code = None
    for _ in range(12):
        try:
            code, res = c.detect(q); break
        except c.RateLimited:
            print("429, cooling 5 min", flush=True); time.sleep(300)
        except c.EgressChanged as e:
            raise SystemExit(str(e))
    if code == 200 and isinstance(res, dict) and "detected" in res:
        done[key] = res["detected"]; LEDGER.write_text(json.dumps(done, indent=0))
        print(f"{key} {res['detected']}  ({len(done)} labelled)", flush=True)
    else:
        print("unexpected", code, flush=True)
print("finished", flush=True)
