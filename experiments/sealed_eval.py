"""One-shot evaluation on the sealed set (voices, texts, and es/fr languages never used anywhere)."""
import glob, json, pathlib, pickle, sys
import numpy as np, soundfile as sf
from sklearn.metrics import roc_auc_score
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
import wavlm_detector2 as wd
ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = json.loads((ROOT / "results/wavlm_detector2_config.json").read_text()); thr = cfg["threshold"]
clf = pickle.load(open(ROOT / "results/wavlm_detector2.pkl", "rb"))
rows = []
for p in sorted(glob.glob(str(ROOT / "audio/sealed/*.wav"))):
    t, v, m = pathlib.Path(p).stem.split("_", 2)
    x, sr = sf.read(p); s = float(np.mean(wd.clip_probs(clf, x, sr)))
    rows.append((t[:2], m, s))
json.dump(rows, open(ROOT / "results/sealed_scores.json", "w"))
y = np.array([int(m in ("sonic-3.6", "sonic-preview")) for _, m, _ in rows]); s = np.array([r[2] for r in rows])
print(f"SEALED: {len(rows)} clips, threshold {thr:.2f}, AUC {roc_auc_score(y, s):.3f}, accuracy {np.mean((s > thr) == y):.3f}")
print(f"  TPR {np.mean(s[y == 1] > thr):.3f}   FPR {np.mean(s[y == 0] > thr):.3f}")
for lang in ["en", "es", "fr"]:
    for m in ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]:
        r = [x[2] for x in rows if x[0] == lang and x[1] == m]
        print(f"  {lang} {m:14s} n={len(r):2d} flagged={np.mean(np.array(r) > thr):.2f} mean={np.mean(r):.2f}")
