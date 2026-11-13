"""Local stand-in detector: logistic regression on loudness-normalised band spectra.

Positives: sonic-3.6 + sonic-preview (watermarked). Negatives: sonic-3.5 + sonic-3.
Evaluated with leave-texts-out CV so it can't memorise content.
"""
import glob, pathlib, sys
import numpy as np, soundfile as sf
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
EDGES = np.geomspace(100, 20000, 61)
MARKED = {"sonic-3.6": 1, "sonic-preview": 1, "sonic-3.5": 0, "sonic-3": 0}


def features(x, sr):
    f, t, Z = signal.stft(x, sr, nperseg=2048)
    P = np.abs(Z) ** 2
    tot = P.sum(0)
    voiced = tot > tot.max() * 1e-4                      # ignore near-silent frames
    Pn = P[:, voiced] / tot[voiced]                      # per-frame loudness normalisation
    return np.array([np.log(Pn[(f >= lo) & (f < hi)].sum(0).mean() + 1e-12) for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def dataset():
    X, y, g, m = [], [], [], []
    for p in sorted(glob.glob(str(ROOT / "audio/corpus2/*.wav"))):
        stem = pathlib.Path(p).stem
        text, voice, model = stem.split("_", 2)
        x, sr = sf.read(p)
        X.append(features(x, sr)); y.append(MARKED[model]); g.append(text); m.append(model)
    return np.array(X), np.array(y), np.array(g), np.array(m)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


if __name__ == "__main__":
    X, y, g, m = dataset()
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(6).split(X, y, g):
        pred[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print(f"leave-texts-out accuracy: {np.mean((pred > .5) == y):.3f}")
    for name in MARKED:
        print(f"  {name:14s} mean P(marked) = {pred[m == name].mean():.2f}")
    # Cross-model generalisation: train on 3.6 vs 3.5 only, test on preview vs 3
    tr = np.isin(m, ["sonic-3.6", "sonic-3.5"]); te = ~tr
    clf = model().fit(X[tr], y[tr]); p2 = clf.predict_proba(X[te])[:, 1]
    print(f"train 3.6/3.5 -> test preview/3 accuracy: {np.mean((p2 > .5) == y[te]):.3f}")
    clf = model().fit(X, y)
    w = clf[-1].coef_[0]
    order = np.argsort(-np.abs(w))[:10]
    print("strongest bands (Hz, weight):", [(f"{EDGES[i]:.0f}-{EDGES[i+1]:.0f}", round(w[i], 2)) for i in sorted(order)])
    import pickle; pickle.dump(clf, open(ROOT / "results/surrogate.pkl", "wb"))
