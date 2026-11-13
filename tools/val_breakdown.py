"""Break validation scores (results/val_scores_<kind>_heldout<TAG>.json from train_head.py) down by language and model.

    python tools/val_breakdown.py conv_heldout_x4c0 [linear_heldout ...]
"""
import collections, json, pathlib, sys
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
MARKED = ("sonic-3.6", "sonic-preview")

for kind in sys.argv[1:]:
    sc = json.loads((ROOT / f"results/val_scores_{kind}.json").read_text())
    def group(k):                                     # keys are "corpus/stem" (older files: bare stems)
        stem = k.split("/")[-1]
        return "en(c3)" if k.startswith("corpus3/") or (stem[0] == "t" and stem[1].isdigit()) else stem.split("_")[0]
    rows = [(group(k), m, p) for k, (m, p) in sc.items()]
    y = np.array([m in MARKED for _, m, _ in rows]); s = np.array([p for *_, p in rows])
    thr = np.sort(s[~y])[int(0.98 * (~y).sum())]
    print(f"== {kind}: {len(rows)} clips, AUC {roc_auc_score(y, s):.3f}, threshold@2%FPR {thr:.3f}, TPR {np.mean(s[y] > thr):.3f}")
    by = collections.defaultdict(list)
    for lang, m, p in rows:
        by[lang].append((m, p))
    print(f"   {'group':8s} {'n':>4s}  " + "  ".join(f"{m:>13s}" for m in ("sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3")) + "   (flag rate at threshold)")
    for lang, v in sorted(by.items()):
        cells = []
        for m in ("sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"):
            ps = [p for mm, p in v if mm == m]
            cells.append(f"{np.mean([p > thr for p in ps]):6.2f} ({len(ps):3d})" if ps else " " * 13)
        print(f"   {lang:8s} {len(v):4d}  " + "  ".join(cells))
    miss = sorted((p, k) for k, (m, p) in sc.items() if m in MARKED and p <= thr)
    fps = sorted(((p, k) for k, (m, p) in sc.items() if m not in MARKED and p > thr), reverse=True)
    print("   misses:", ", ".join(f"{k}={p:.2f}" for p, k in miss[:15]))
    print("   false positives:", ", ".join(f"{k}={p:.2f}" for p, k in fps[:10]))
