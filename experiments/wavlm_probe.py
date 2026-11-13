"""Per-layer linear probes on WavLM features: does any layer carry a *model-independent* mark signal?
A: train sonic-3.6 vs 3.5 -> test sonic-preview vs 3 (held-out voices)
B: train sonic-preview vs 3 -> test sonic-3.6 vs 3.5 (held-out voices)
C: train all four -> held-out voices, per-model flag rates
Held-out voices: corpus3 v24-v29 (never used for training here)."""
import sys, pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ROOT = pathlib.Path(__file__).resolve().parent.parent
d = np.load(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/wavlm_feats.npz"))
keys, F = d["keys"], d["feats"]
meta = [pathlib.Path(k).stem.split("_", 2) for k in keys]
model = np.array([m for _, _, m in meta]); corpus = np.array([k.split("/")[1] for k in keys])
voice = np.array([v for _, v, _ in meta])
held = (corpus == "corpus3") & np.array([v.startswith("v") and int(v[1:]) >= 24 for v in voice])
y = np.isin(model, ["sonic-3.6", "sonic-preview"]).astype(int)


def fit(tr, Xl):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(Xl[tr], y[tr])


print(f"{len(keys)} clips, {held.sum()} held-out")
print("layer   A:3.6/3.5->preview/3   B:preview/3->3.6/3.5   C:all->held-out AUC  (flag rates 3.6/prev/3.5/3)")
for L in range(F.shape[1]):
    X = F[:, L]
    out = []
    for trm, tem in [(("sonic-3.6", "sonic-3.5"), ("sonic-preview", "sonic-3")), (("sonic-preview", "sonic-3"), ("sonic-3.6", "sonic-3.5"))]:
        tr = ~held & np.isin(model, trm); te = held & np.isin(model, tem)
        out.append(roc_auc_score(y[te], fit(tr, X).predict_proba(X[te])[:, 1]))
    clf = fit(~held, X); p = clf.predict_proba(X[held])[:, 1]
    rates = "/".join(f"{np.mean(p[model[held] == m] > .5):.2f}" for m in ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"])
    print(f"  {L:2d}        {out[0]:.3f}                  {out[1]:.3f}                  {roc_auc_score(y[held], p):.3f}   ({rates})")
