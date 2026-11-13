"""Choose the final ensemble and its threshold on the validation voices only (never a sealed set).

    python tools/calibrate_final.py <config_out.json> <candidateA: m1,m2,...> <candidateB: m1,m2,...>

Members are final_detector names (e.g. r3ls0 -> results/head_conv_heldout_r3ls0.pt, finetune_r2 -> fine-tuned).
Rule fixed before scoring the sealed set: pick the candidate with the higher mean TPR at 1/2/5% FPR on the
validation voices (ties -> A); its threshold is the 2%-FPR point on validation.
Writes the config and <config_out>_val_members.json (per-clip member logits).
(Round 2 used an earlier version of this script: 3 frozen heads vs 3 frozen heads + fine-tuned.)
"""
import glob, json, pathlib, sys
import numpy as np, soundfile as sf
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train_head as th
from final_detector import Detector

ROOT = th.ROOT
CONFIG = ROOT / sys.argv[1]
A, B = sys.argv[2].split(","), sys.argv[3].split(",")
members = list(dict.fromkeys(A + B))
dets = {m: Detector(members=[m], config="/nonexistent") for m in members}
rows = {}
for corpus in ("corpus3", "corpus5", "corpus6", "corpus7"):
    for p in sorted(glob.glob(str(ROOT / f"audio/{corpus}/*.wav"))):
        m = th.meta(p)
        if th.excluded(m) or not th.is_val(m) or (corpus == "corpus3" and m["stem"].split("_")[1] in ("v01", "v05")):
            continue
        x, sr = sf.read(p)
        rows[f"{corpus}/{m['stem']}"] = (m["model"], {k: d.member_logits(x, sr)[0] for k, d in dets.items()})
CONFIG.with_name(CONFIG.stem + "_val_members.json").write_text(json.dumps(rows))

y = np.array([m in th.MARKED for m, _ in rows.values()])


def ens(names):
    return np.array([np.mean([l[n] for n in names]) for _, l in rows.values()])


def report(name, s):
    neg = np.sort(s[~y]); t = {f: float(np.mean(s[y] > neg[int((1 - f) * len(neg))])) for f in (0.01, 0.02, 0.05)}
    print(f"{name:34s} AUC {roc_auc_score(y, s):.3f}  TPR@1/2/5% {t[0.01]:.3f}/{t[0.02]:.3f}/{t[0.05]:.3f}", flush=True)
    return np.mean(list(t.values()))


print(f"validation: {len(y)} clips ({y.sum()} marked)")
for n in members:
    report(n, ens([n]))
en = np.array([k.startswith("corpus3/") or k.split("/")[1].startswith("en_") for k in rows])
for name, sel in (("English", en), ("non-English", ~en)):
    s_ = ens(A); print(f"  A on {name} val: AUC {roc_auc_score(y[sel], s_[sel]):.3f}", end="")
    s_ = ens(B); print(f"  | B: AUC {roc_auc_score(y[sel], s_[sel]):.3f}  (n={sel.sum()})")
qa = report("A = " + "+".join(A), ens(A)); qb = report("B = " + "+".join(B), ens(B))
chosen = A if qa >= qb else B
s = ens(chosen); neg = np.sort(s[~y]); thr = float(neg[int(0.98 * len(neg))])
CONFIG.write_text(json.dumps({"members": chosen, "feat": th.FEAT, "threshold": thr, "val_clips": int(len(y)),
                              "val_tpr": float(np.mean(s[y] > thr)), "val_fpr": float(np.mean(s[~y] > thr))}, indent=1))
print(f"chosen: {chosen}, threshold {thr:.3f} (val TPR {np.mean(s[y] > thr):.3f}, FPR {np.mean(s[~y] > thr):.3f})")
