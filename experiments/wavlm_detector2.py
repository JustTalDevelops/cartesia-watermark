"""Windowed Cartesia-watermark detector: WavLM layer-1 window features + logistic regression.
Training excludes held-out voices (corpus3 v24-v29) and validation voices (corpus3 v01, v05)."""
import json, pathlib, pickle, sys
import numpy as np, soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
from wavlm_windows import windows

ROOT = pathlib.Path(__file__).resolve().parent.parent
import wavlm_windows
SUFFIX = "_band" if wavlm_windows.BAND else ""
POOLS = {"max": np.max, "top2": lambda p: np.sort(p)[-2:].mean(), "mean": np.mean}


def excluded(key):
    parts = key.split("/")
    if parts[1] != "corpus3":
        return False
    v = pathlib.Path(key).stem.split("_", 2)[1]
    return int(v[1:]) >= 24 or int(v[1:]) in (1, 5)


def heldout(key):
    return key.split("/")[1] == "corpus3" and int(pathlib.Path(key).stem.split("_", 2)[1][1:]) >= 24


def train():
    d = np.load(ROOT / f"results/wavlm_windows{SUFFIX}.npz")
    keep = np.array([not excluded(k) for k in d["keys"]])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))
    clf.fit(d["feats"][keep], d["label"][keep])
    pickle.dump(clf, open(ROOT / f"results/wavlm_detector2{SUFFIX}.pkl", "wb"))
    print(f"trained on {keep.sum()} windows", flush=True)
    return clf, d


def clip_probs(clf, x, sr):
    w = windows(x, sr)
    return clf.predict_proba(np.array(w))[:, 1] if w else np.array([0.0])


if __name__ == "__main__":
    clf, d = train()
    # 1. held-out natural clips (clean windows already extracted)
    print("== held-out voices, natural clips")
    for pool, fn in POOLS.items():
        sc = {}
        keys = d["keys"]; m = np.array([heldout(k) for k in keys]) & (d["variant"] == "clean")
        P = clf.predict_proba(d["feats"][m])[:, 1]
        for k, p in zip(keys[m], P):
            sc.setdefault(k, []).append(p)
        ks = list(sc); s = np.array([fn(np.array(sc[k])) for k in ks])
        y = np.array([int(pathlib.Path(k).stem.split("_", 2)[2] in ("sonic-3.6", "sonic-preview")) for k in ks])
        mdl = np.array([pathlib.Path(k).stem.split("_", 2)[2] for k in ks])
        rates = " ".join(f"{mm}={np.mean(s[mdl == mm] > .5):.2f}" for mm in ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"])
        print(f"  {pool:5s} AUC {roc_auc_score(y, s):.3f}  flag rates: {rates}")
    # 2./3. Cartesia-labelled sets
    import validate
    items = [(n, x, sr, lab, kind) for n, x, sr, lab, kind in validate.items()]
    probs = {n: clip_probs(clf, x, sr) for n, x, sr, _, _ in items}
    dist = json.loads((ROOT / "results/distill.json").read_text())
    dprobs = {}
    for k, v in dist.items():
        x, sr = sf.read(ROOT / v["path"]); dprobs[k] = (v.get("family", "natural"), v["detected"], clip_probs(clf, x, sr))
    for pool, fn in POOLS.items():
        rows = [(n, kind, lab, fn(probs[n])) for n, x, sr, lab, kind in items]
        core = [r for r in rows if r[1] != "fp"]
        agree = {kind: np.mean([(r[3] > .5) == r[2] for r in rows if r[1] == kind]) for kind in ["marked", "unmarked", "fp"]}
        da = [(fam, y, fn(p)) for fam, y, p in dprobs.values()]
        fams = " ".join(f"{f}={np.mean([(p > .5) == y for ff, y, p in da if ff == f]):.2f}" for f in sorted(set(x[0] for x in da)))
        print(f"== pool={pool}: probe set {np.mean([(r[3] > .5) == r[2] for r in core]):.3f} "
              f"(marked {agree['marked']:.2f}, unmarked {agree['unmarked']:.2f}, cartesia-FP {agree['fp']:.2f}) | "
              f"distill {np.mean([(p > .5) == y for _, y, p in da]):.3f} on {len(da)}\n     distill by family: {fams}")
        if pool == "max":
            with open(ROOT / f"results/wavlm2{SUFFIX}_validate.txt", "w") as f:
                for n, kind, lab, p in rows:
                    f.write(f"{n:34s} {kind:9s} cartesia={lab} local={p:.2f} {'OK' if (p > .5) == lab else 'XX'}\n")
