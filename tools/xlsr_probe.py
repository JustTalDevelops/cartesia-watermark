"""Quick layer probe: does a multilingual SSL model (XLS-R 300M) beat English-only WavLM on non-English clips?

    python tools/xlsr_probe.py extract <shard> <nshards> <threads>   # per-clip mean+std of each low layer
    python tools/xlsr_probe.py fit                                   # logistic regression per (model, layer)

Clean clips only, same voice-disjoint split as train_head.py; validation AUC reported separately for English and
non-English validation voices.
"""
import glob, pathlib, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train_head as th
from frames import to16k

ROOT = th.ROOT
OUT = ROOT / "features/probe_layers"
SPECS = {"xlsr": ("facebook/wav2vec2-xls-r-300m", 7), "wavlm": ("microsoft/wavlm-base-plus", 4)}   # model, layers kept


def clips():
    out = []
    for corpus in th.TRAIN_CORPORA:
        for p in sorted(glob.glob(str(ROOT / f"audio/{corpus}/*.wav"))):
            m = th.meta(p)
            if m["corpus"] == "corpus3" and th.is_val(m) and m["stem"].split("_")[1] in ("v01", "v05"):
                continue
            out.append((p, m))
    return out


def lang(m):
    return "en" if m["corpus"] in ("corpus3", "corpus4") else m["stem"].split("_")[0][:2]


def extract(shard, nshards, threads):
    from transformers import AutoModel
    torch.set_num_threads(threads)
    models = {}
    for k, (name, n) in SPECS.items():
        mm = AutoModel.from_pretrained(name).eval(); mm.encoder.layers = mm.encoder.layers[:n - 1]; models[k] = mm
    todo = clips()[shard::nshards]
    for i, (p, m) in enumerate(todo):
        out = OUT / m["corpus"] / f"{m['stem']}.npz"
        if out.exists():
            continue
        x, sr = sf.read(p); y = to16k(x if x.ndim == 1 else x.mean(1), sr)
        yn = torch.tensor(((y - y.mean()) / (y.std() + 1e-9)).astype(np.float32))[None]
        feats = {}
        with torch.no_grad():
            for k, mm in models.items():
                hs = mm(yn, output_hidden_states=True).hidden_states
                for li, h in enumerate(hs):
                    h = h[0].numpy(); feats[f"{k}{li}"] = np.concatenate([h.mean(0), h.std(0)]).astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True); np.savez(out, **feats)
        if i % 200 == 0:
            print(f"shard {shard}: {i}/{len(todo)}", flush=True)
    print(f"shard {shard} done", flush=True)


def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = [(m, np.load(OUT / m["corpus"] / f"{m['stem']}.npz")) for _, m in clips() if (OUT / m["corpus"] / f"{m['stem']}.npz").exists()]
    keys = list(rows[0][1].keys())
    tr = [i for i, (m, _) in enumerate(rows) if not th.is_val(m)]; va = [i for i, (m, _) in enumerate(rows) if th.is_val(m)]
    y = np.array([m["model"] in th.MARKED for m, _ in rows])
    en = np.array([lang(m) == "en" for m, _ in rows])
    print(f"{len(tr)} train / {len(va)} val clips ({en[va].sum()} English val)")
    for k in keys:
        X = np.array([f[k] for _, f in rows])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(X[tr], y[tr])
        s = clf.decision_function(X[va]); yv = y[va]; ev = en[va]
        print(f"  {k:7s} val AUC all {roc_auc_score(yv, s):.3f}  English {roc_auc_score(yv[ev], s[ev]):.3f}  "
              f"non-English {roc_auc_score(yv[~ev], s[~ev]):.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        fit()
