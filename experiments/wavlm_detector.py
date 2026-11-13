"""Cartesia-watermark detector: WavLM-base-plus layer-1 features (mean+std pooled) + logistic regression.

Train: all natural corpus3/corpus4 clips except held-out voices (corpus3 v24-v29) and the validation voices
(corpus3 v01 Daniel, v05 Jacqueline). Label = model is sonic-3.6 / sonic-preview.
"""
import json, pathlib, pickle, sys
import numpy as np, soundfile as sf, torch
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import WavLMModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = 1
_wavlm = None


def wavlm():
    global _wavlm
    if _wavlm is None:
        torch.set_num_threads(12)
        _wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
    return _wavlm


def featurize(x, sr):
    """Audio (any rate) -> layer-1 mean+std features. Same preprocessing as wavlm_feats.py."""
    x = np.asarray(x, np.float64)
    from math import gcd
    g = gcd(int(sr), 16000)
    y = signal.resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-9)
    with torch.no_grad():
        h = wavlm()(torch.tensor(y)[None], output_hidden_states=True).hidden_states[LAYER][0]
    return np.concatenate([h.mean(0).numpy(), h.std(0).numpy()])


def train():
    d = np.load(ROOT / "results/wavlm_feats.npz")
    keys, F = d["keys"], d["feats"][:, LAYER]
    meta = [pathlib.Path(k).stem.split("_", 2) for k in keys]
    corpus = np.array([k.split("/")[1] for k in keys])
    voice = np.array([v for _, v, _ in meta]); model = np.array([m for _, _, m in meta])
    excl = (corpus == "corpus3") & np.array([v.startswith("v") and (int(v[1:]) >= 24 or int(v[1:]) in (1, 5)) for v in voice])
    y = np.isin(model, ["sonic-3.6", "sonic-preview"]).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)).fit(F[~excl], y[~excl])
    pickle.dump(clf, open(ROOT / "results/wavlm_detector.pkl", "wb"))
    print(f"trained on {int((~excl).sum())} clips", flush=True)
    return clf


def load():
    return pickle.load(open(ROOT / "results/wavlm_detector.pkl", "rb"))


def score(clf, x, sr):
    return float(clf.predict_proba(featurize(x, sr)[None])[0, 1])


if __name__ == "__main__":
    clf = train()
    sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
    import validate
    print("== 1. Cartesia probe set (121 items)", flush=True)
    rows = []
    for name, x, sr, lab, kind in validate.items():
        p = score(clf, x, sr); rows.append((name, kind, lab, p))
    for kind in ["marked", "unmarked", "fp"]:
        r = [x for x in rows if x[1] == kind]
        print(f"  agreement on {kind:9s} ({len(r):2d}): {np.mean([(x[3] > .5) == x[2] for x in r]):.2f}")
    core = [x for x in rows if x[1] != "fp"]
    print(f"  OVERALL (excl. Cartesia FPs): {np.mean([(x[3] > .5) == x[2] for x in core]):.3f}")
    with open(ROOT / "results/wavlm_validate.txt", "w") as f:
        for name, kind, lab, p in rows:
            f.write(f"{name:34s} {kind:9s} cartesia={lab} local={p:.2f} {'OK' if (p > .5) == lab else 'XX'}\n")
    print("== 2. Cartesia distillation labels (never trained on)", flush=True)
    d = json.loads((ROOT / "results/distill.json").read_text())
    out = []
    for k, v in d.items():
        x, sr = sf.read(ROOT / v["path"]); out.append((v.get("family", "natural"), v["detected"], score(clf, x, sr)))
    print(f"  agreement {np.mean([(p > .5) == y for _, y, p in out]):.3f} on {len(out)} labels")
    for fam in sorted(set(o[0] for o in out)):
        r = [o for o in out if o[0] == fam]
        print(f"    {fam:10s} n={len(r):3d} agree={np.mean([(p > .5) == y for _, y, p in r]):.2f}")
