"""Train detector heads on frozen SSL frames (from tools/frames.py and tools/frames_contrast.py).

    [FEAT=xlsr2] [CONTRAST=1] [SEED=0] [SMOOTH=0.1] [TAG=_x4c0] python tools/train_head.py {linear|conv} {heldout|lmo|final} [epochs]

Environment
  FEAT     feature set: wavlm1 (WavLM layer 1, features/) or xlsr2 (XLS-R layer 2, features_xlsr2/)
  CONTRAST 1 = also train on the mark-targeted variants (delay10 -> marked; delay40, twinband -> not marked)
  SEED, SMOOTH (label smoothing), TAG (suffix for saved files), THREADS
  HOLDOUT  1 = drop one English sentence from training (text-memorisation diagnostic, see held_text)

Splits (validation = corpus3 v24-v29 plus every 8th corpus5/6/7 voice id; split by voice id, never by clip)
  heldout : train on everything else; report validation AUC/TPR, the 121 Cartesia probe items and the
            Cartesia distillation labels. Saves results/head_<kind>_heldout<TAG>.pt and per-clip validation scores.
  lmo     : leave-model-out confound check: train on sonic-3.6 vs 3.5 only, test on preview vs 3.
  final   : same training data as heldout.
"""
import glob, json, pathlib, pickle, sys
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
WIN, HOP, GATE = 150, 75, 10 ** (-50 / 20)
MARKED = ("sonic-3.6", "sonic-preview")
import os
FEAT = os.environ.get("FEAT", "wavlm1")                           # feature set (see frames.py)
FEATROOT = ROOT / ("features" if FEAT == "wavlm1" else f"features_{FEAT}")
DIM = {"wavlm1": 768, "xlsr2": 1024}[FEAT]
CONTRAST = os.environ.get("CONTRAST", "") == "1"                   # include tools/frames_contrast.py variants
SEED = int(os.environ.get("SEED", 0)); SMOOTH = float(os.environ.get("SMOOTH", 0)); TAG = os.environ.get("TAG", "")
torch.set_num_threads(int(os.environ.get("THREADS", 8))); torch.manual_seed(SEED); np.random.seed(SEED)


def meta(path):
    p = pathlib.Path(path); corpus = p.parent.name; variant = p.parent.parent.name
    stem = p.stem; model = next(m for m in ("sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3") if stem.endswith(m))
    return dict(corpus=corpus, variant=variant, stem=stem, model=model)


# corpus5/6 rows -> voice id. Multilingual voices appear under several languages, so splits go by voice id.
# Round 2 dropped corpus5 rows whose voice was in sealed2. Sealed2 is now spent (scored once on 2026-11-12), so
# round 3 trains on it and on the old sealed set; sealed3 voices are disjoint from all of these by construction.
_PLAN = __import__("json").loads((ROOT / "audio/corpus5_plan.json").read_text())
C5_VOICE = {i: r[2] for i, r in enumerate(_PLAN)}
SEALED2_VOICES = {r[2] for r in _PLAN if r[0] == "sealed2"}
_C5_VOICES = sorted({r[2] for r in _PLAN if r[0] == "corpus5"} - SEALED2_VOICES)
C5_VAL_VOICES = set(_C5_VOICES[::8])                                # unchanged from round 2
_PLAN6 = __import__("json").loads((ROOT / "audio/corpus6_plan.json").read_text())
C6_VOICE = {i: r[2] for i, r in enumerate(_PLAN6)}
C6_VAL_VOICES = set(sorted({r[2] for r in _PLAN6 if r[0] == "corpus6"})[::8])
_P7 = ROOT / "audio/corpus7_plan.json"
_PLAN7 = __import__("json").loads(_P7.read_text()) if _P7.exists() else []
C7_VOICE = {i: r[2] for i, r in enumerate(_PLAN7)}
C7_VAL_VOICES = set(sorted({r[2] for r in _PLAN7 if r[0] == "corpus7"})[::8])
# sealed3 was scored once (2026-11-12, round 3) and is now training data too; sealed4 is the round-4 test set.
TRAIN_CORPORA = ("corpus3", "corpus4", "corpus5", "corpus6", "corpus7", "sealed", "sealed2", "sealed3")


HOLD_TEXT = "She kept every letter her grandfather wrote"           # text-holdout diagnostic (env HOLDOUT=1)


def held_text(m):
    """True for clips of the held-out English sentence (index 5 in corpus3/corpus4, that sentence in corpus5/6)."""
    c, stem = m["corpus"], m["stem"]
    if c == "corpus3":
        return stem.startswith("t5_")
    if c == "corpus4":
        return stem.startswith("en5_")
    if c in ("corpus5", "corpus6"):
        plan = _PLAN if c == "corpus5" else _PLAN6
        return plan[int(stem.split("_")[1])][3].startswith(HOLD_TEXT)
    return False


def excluded(m):
    return False                                                    # round 3: nothing to drop (see above)


def is_val(m):
    """Validation voices: corpus3 v24-v29 (+ v01/v05, used by the probe set) and every 8th corpus5 voice id."""
    if m["corpus"] == "corpus5":
        return C5_VOICE[int(m["stem"].split("_")[1])] in C5_VAL_VOICES
    if m["corpus"] == "corpus6":
        return C6_VOICE[int(m["stem"].split("_")[1])] in C6_VAL_VOICES
    if m["corpus"] == "corpus7":
        return C7_VOICE[int(m["stem"].split("_")[1])] in C7_VAL_VOICES
    if m["corpus"] != "corpus3":
        return False
    v = int(m["stem"].split("_")[1][1:])
    return v >= 24 or v in (1, 5)


def windows_of(h, rms):
    out = []
    for s in range(0, max(1, len(h) - WIN + 1), HOP):
        if rms[s:s + WIN].mean() < GATE:
            continue
        w = h[s:s + WIN]
        if len(w) < WIN:
            w = np.pad(w, ((0, WIN - len(w)), (0, 0)))
        out.append(w)
    return out


def load(split):
    X, Y, G, W = [], [], [], []               # windows, labels, group (clip key), weight
    for f in sorted(glob.glob(str(FEATROOT / "*/*/*.npz"))):
        m = meta(f)
        if m["corpus"] not in TRAIN_CORPORA or excluded(m):
            continue
        if m["variant"] in ("delay10", "delay40", "twinband") and not CONTRAST:
            continue
        if os.environ.get("HOLDOUT") == "1" and held_text(m):
            continue
        val = is_val(m)
        if split in ("heldout", "final") and val:
            continue
        if split == "lmo" and (val or m["model"] not in ("sonic-3.6", "sonic-3.5")):
            continue
        d = np.load(f)
        lab = 0 if m["variant"] in ("reversed", "delay40", "twinband") else int(m["model"] in MARKED)
        for w in windows_of(d["h"], d["rms"]):
            X.append(w); Y.append(lab); G.append(m["stem"]); W.append(2.0 if m["model"] == "sonic-3.5" else 1.0)
    return np.array(X, dtype=np.float16), np.array(Y), np.array(G), np.array(W, dtype=np.float32)


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.net = nn.Sequential(nn.Conv1d(DIM, 256, 5, padding=2), nn.GELU(), nn.Dropout(0.2),
                                 nn.Conv1d(256, 256, 5, padding=4, dilation=2), nn.GELU(), nn.Dropout(0.2))
        self.att = nn.Conv1d(256, 1, 1); self.out = nn.Linear(256, 1)

    def forward(self, x):                      # x: B x T x 768
        z = self.net(self.norm(x).transpose(1, 2))
        a = torch.softmax(self.att(z), -1)
        return self.out((z * a).sum(-1))[:, 0]


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(2 * DIM, affine=False); self.out = nn.Linear(2 * DIM, 1)

    def forward(self, x):
        return self.out(self.norm(torch.cat([x.mean(1), x.std(1)], 1)))[:, 0]


def fit(kind, X, Y, epochs, log=print, W=None):
    net = Conv() if kind == "conv" else Linear()
    opt = torch.optim.AdamW(net.parameters(), 1e-3 if kind == "conv" else 3e-3, weight_decay=5e-2)
    n = len(X); steps = epochs * ((n + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3 if kind == "conv" else 3e-3, total_steps=steps)
    Xt = torch.from_numpy(X); Yt = torch.tensor(Y, dtype=torch.float32)
    pos_w = torch.tensor((len(Y) - Y.sum()) / max(1, Y.sum()), dtype=torch.float32)
    Wt = torch.tensor(W if W is not None else np.ones(len(Y), np.float32))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n); tot = 0
        for i in range(0, n, 128):
            b = perm[i:i + 128]; xb = Xt[b].float()
            xb = xb + 0.05 * torch.randn_like(xb) * xb.std()            # light feature noise
            yb = Yt[b] * (1 - SMOOTH) + 0.5 * SMOOTH                      # optional label smoothing
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb), yb, weight=Wt[b], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        log(f"  epoch {ep + 1:2d} loss {tot / n:.3f}")
    return net.eval()


@torch.no_grad()
def predict(net, wins):
    if not len(wins):
        return np.array([0.0])
    return torch.sigmoid(net(torch.tensor(np.array(wins)).float())).numpy()


def eval_heldout(net, models=("sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"), label=""):
    sc = {}
    for f in sorted(glob.glob(str(FEATROOT / "clean/corpus3/*.npz")) + glob.glob(str(FEATROOT / "clean/corpus5/*.npz")) + glob.glob(str(FEATROOT / "clean/corpus6/*.npz")) + glob.glob(str(FEATROOT / "clean/corpus7/*.npz"))):
        m = meta(f)
        if excluded(m) or not is_val(m) or m["model"] not in models or m["stem"].split("_")[1] in ("v01", "v05"):
            continue
        d = np.load(f); sc[f'{m["corpus"]}/{m["stem"]}'] = (m["model"], float(np.mean(predict(net, windows_of(d["h"], d["rms"])))))
    y = np.array([int(v[0] in MARKED) for v in sc.values()]); s = np.array([v[1] for v in sc.values()])
    rates = " ".join(f"{mm}={np.mean([v[1] > .5 for v in sc.values() if v[0] == mm]):.2f}" for mm in models)
    print(f"  {label} validation voices ({len(y)} clips): AUC {roc_auc_score(y, s):.3f}  flag@0.5: {rates}", flush=True)
    if y.min() == 0 and y.max() == 1:
        thr = sorted(s[y == 0])[int(0.98 * (y == 0).sum())]                # threshold at 2% FPR
        print(f"    @2% FPR: threshold {thr:.3f}, TPR {np.mean(s[y == 1] > thr):.3f}", flush=True)
    if label == "":
        (ROOT / f"results/val_scores_{sys.argv[1]}_{sys.argv[2]}{TAG}.json").write_text(json.dumps(sc))
    return sc


def eval_cartesia(net):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import validate, frames as fr, soundfile as sf
    def score(x, sr):
        h, rms = fr.frames(fr.to16k(x, sr)); return float(np.mean(predict(net, windows_of(h, rms))))
    rows = [(n, kind, lab, score(x, sr)) for n, x, sr, lab, kind in validate.items()]
    core = [r for r in rows if r[1] != "fp"]
    res = {k: np.mean([(r[3] > .5) == r[2] for r in rows if r[1] == k]) for k in ("marked", "unmarked")}
    d = json.loads((ROOT / "results/distill.json").read_text())
    da = []
    for k, v in d.items():
        x, sr = sf.read(ROOT / v["path"]); da.append((v.get("family", "natural"), v["detected"], score(x, sr)))
    print(f"  probe set {np.mean([(r[3] > .5) == r[2] for r in core]):.3f} (marked {res['marked']:.2f}, unmarked {res['unmarked']:.2f})"
          f" | Cartesia labels {np.mean([(p > .5) == y for _, y, p in da]):.3f} on {len(da)}"
          f" (natural {np.mean([(p > .5) == y for f, y, p in da if f == 'natural']):.2f})", flush=True)
    return rows, da


if __name__ == "__main__":
    kind, split = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    X, Y, G, W = load(split)
    print(f"[{kind}/{split}] {len(X)} windows ({int(Y.sum())} marked) from {len(set(G))} clips", flush=True)
    net = fit(kind, X, Y, epochs, log=lambda s: print(s, flush=True), W=W)
    if split == "lmo":
        eval_heldout(net, models=("sonic-preview", "sonic-3"), label="UNSEEN MODELS")
    else:
        eval_heldout(net)
        eval_cartesia(net)
    torch.save(net.state_dict(), ROOT / f"results/head_{kind}_{split}{TAG}.pt")
