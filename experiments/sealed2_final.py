"""Score sealed2 ONCE with the round-2 detector (results/final_detector_r2.json) and compare with Cartesia's own detector.

Ground truth is provenance: sonic-3.6 / sonic-preview are marked, sonic-3.5 / sonic-3 are not.
Refuses to run twice (results/sealed2_final.json is the record).
"""
import collections, json, pathlib, sys
import numpy as np, soundfile as sf
from sklearn.metrics import roc_auc_score
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
import train_head as th
from final_detector import Detector

ROOT = th.ROOT
OUT = ROOT / "results/sealed2_final.json"
if OUT.exists():
    raise SystemExit("sealed2 has already been scored; see results/sealed2_final.json")

det = Detector(config=ROOT / "results/final_detector_r2.json")   # the round-2 model that was scored
cart = json.loads((ROOT / "results/sealed2_cartesia.json").read_text())
rows = []
for p in sorted((ROOT / "audio/sealed2").glob("*.wav")):
    x, sr = sf.read(p); m = th.meta(p)["model"]; lang = p.name.split("_")[0]
    s = det.score(x, sr)
    rows.append(dict(clip=p.name, lang=lang, model=m, marked=m in th.MARKED, score=s, ours=s is not None and s > det.threshold,
                     cartesia=cart.get(p.name), seconds=len(x) / sr))
OUT.write_text(json.dumps({"threshold": det.threshold, "rows": rows}, indent=0))

y = np.array([r["marked"] for r in rows]); s = np.array([r["score"] if r["score"] is not None else -99 for r in rows])
ours = np.array([r["ours"] for r in rows]); ca = np.array([bool(r["cartesia"]) for r in rows])
print(f"sealed2: {len(rows)} clips, {y.sum()} marked; threshold {det.threshold:.3f} (fixed on validation)")
print(f"  ours     : AUC {roc_auc_score(y, s):.3f}  TPR {ours[y].mean():.3f}  FPR {ours[~y].mean():.3f}  accuracy {(ours == y).mean():.3f}")
print(f"  Cartesia : TPR {ca[y].mean():.3f}  FPR {ca[~y].mean():.3f}  accuracy {(ca == y).mean():.3f}")
print(f"  agreement ours vs Cartesia: {(ours == ca).mean():.3f}")
neg = np.sort(s[~y])
for f in (0.02, 0.05, 0.10):
    print(f"  (threshold-free) TPR at {int(f * 100)}% FPR on sealed2 itself: {np.mean(s[y] > neg[int((1 - f) * len(neg))]):.3f}")
print("  per model (flag rate ours / Cartesia):")
for m in ("sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"):
    i = np.array([r["model"] == m for r in rows]); print(f"    {m:14s} {ours[i].mean():.2f} / {ca[i].mean():.2f}  (n={i.sum()})")
print("  per language (marked caught ours/Cartesia, unmarked flagged ours/Cartesia):")
seen = {"en", "es", "fr", "de", "ja", "hi"}
for lang in sorted({r["lang"] for r in rows}):
    i = np.array([r["lang"] == lang for r in rows])
    print(f"    {lang}{'' if lang in seen else '*'}  {ours[i & y].sum()}/{(i & y).sum()} vs {ca[i & y].sum()}/{(i & y).sum()}   "
          f"{ours[i & ~y].sum()}/{(i & ~y).sum()} vs {ca[i & ~y].sum()}/{(i & ~y).sum()}")
print("  (* = language never seen in training)")
