"""Fine-tune WavLM-base-plus (CNN encoder + layer 1) together with the conv head, on raw 16 kHz audio.

    python tools/finetune.py [epochs] [init_head]

Used in round 2 (WavLM only). Starts from a head trained on frozen features (default results/head_conv_heldout_ls0.pt),
so epoch 0 is approximately that model (per-window instead of per-clip normalisation).
Same voice-disjoint split as train_head.py. Each epoch draws one random 3 s speech window per clip, from the clean,
noisy (17-30 dB SNR) or mp3 (32/64 kbps) version; time-reversed marked clips are extra negatives, and
sonic-3.5 counts double. After every epoch it scores the validation voices and keeps the best checkpoint
(highest mean TPR at 1/2/5% FPR) in results/finetune{TAG}_best.pt, with per-clip scores in results/val_scores_finetune{TAG}.json.
"""
import glob, json, os, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import WavLMModel
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train_head as th
from frames import to16k
import subprocess, tempfile

ROOT = th.ROOT
SEG, HOP16, GATE = 48000, 24000, th.GATE
torch.set_num_threads(int(os.environ.get("THREADS", 20))); torch.manual_seed(0)
rng = np.random.default_rng(0)
TAG = os.environ.get("TAG", "")                                    # output name suffix, e.g. _r3


class Net(nn.Module):
    def __init__(self, head_state):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm.encoder.layers = self.wavlm.encoder.layers[:1]
        self.wavlm.config.mask_time_prob = 0.0                      # no SpecAugment-style masking
        self.head = th.Conv(); self.head.load_state_dict(head_state)

    def forward(self, x):                                          # x: B x SEG (per-window z-normalised)
        return self.head(self.wavlm(x, output_hidden_states=True).hidden_states[1])


def znorm(w):
    return ((w - w.mean()) / (w.std() + 1e-9)).astype(np.float32)


def speech_starts(y):
    """Window starts whose mean frame RMS is above the gate (same rule as the frozen-feature windows)."""
    starts = []
    for s in range(0, max(1, len(y) - SEG + 1), HOP16):
        if np.sqrt(np.mean(y[s:s + SEG] ** 2)) >= GATE:
            starts.append(s)
    return starts or [0]


def prep(path):
    x, sr = sf.read(path); x = x if x.ndim == 1 else x.mean(1)
    y = to16k(x, sr)
    k = int(rng.choice([32, 64]))
    ym = to16k(mp3(x, sr, k), sr)[:len(y)]
    return path, y.astype(np.float32), ym.astype(np.float32)


def mp3(x, sr, kbps):
    """mp3 round trip; single-threaded ffmpeg with retries (many parallel ffmpegs hit EAGAIN)."""
    for attempt in range(5):
        try:
            with tempfile.TemporaryDirectory() as d:
                sf.write(f"{d}/a.wav", x, sr, subtype="FLOAT")
                subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-y", "-i", f"{d}/a.wav", "-b:a", f"{kbps}k", f"{d}/b.mp3"], check=True)
                subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-y", "-i", f"{d}/b.mp3", "-ar", str(sr), "-ac", "1", f"{d}/c.wav"], check=True)
                return sf.read(f"{d}/c.wav")[0]
        except subprocess.CalledProcessError:
            time.sleep(1 + attempt)
    raise RuntimeError("ffmpeg failed")


def noisy(y):
    snr = rng.uniform(17, 30)
    return y + rng.standard_normal(len(y)).astype(np.float32) * np.sqrt(np.mean(y ** 2) / 10 ** (snr / 10))


def window(y, s):
    w = y[s:s + SEG]
    return np.pad(w, (0, SEG - len(w))) if len(w) < SEG else w


@torch.no_grad()
def score_clip(net, y):
    ws = [znorm(window(y, s)) for s in speech_starts(y)]
    return float(torch.sigmoid(net(torch.tensor(np.array(ws)))).mean())


def evaluate(net, val):
    net.eval()
    sc = {stem: (model, score_clip(net, y)) for stem, model, y in val}
    yv = np.array([m in th.MARKED for m, _ in sc.values()]); s = np.array([p for _, p in sc.values()])
    neg = np.sort(s[~yv]); tpr = {f: float(np.mean(s[yv] > neg[int((1 - f) * len(neg))])) for f in (0.01, 0.02, 0.05)}
    return sc, roc_auc_score(yv, s), tpr


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    init = sys.argv[2] if len(sys.argv) > 2 else "results/head_conv_heldout_ls0.pt"
    train, val = [], []
    for corpus in th.TRAIN_CORPORA:
        for p in sorted(glob.glob(str(ROOT / f"audio/{corpus}/*.wav"))):
            m = th.meta(p)
            if th.excluded(m):
                continue
            if th.is_val(m):
                if not (m["corpus"] == "corpus3" and m["stem"].split("_")[1] in ("v01", "v05")):
                    val.append(p)
            else:
                train.append(p)
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        tr = list(ex.map(prep, train, chunksize=8))
        va = list(ex.map(prep, val, chunksize=8))
    print(f"loaded {len(tr)} train / {len(va)} val clips in {time.time() - t0:.0f}s", flush=True)
    items = []                                                     # (clean, mp3, label, weight, starts)
    for p, y, ym in tr:
        m = th.meta(p); lab = int(m["model"] in th.MARKED)
        items.append((y, ym, lab, 2.0 if m["model"] == "sonic-3.5" else 1.0, speech_starts(y), False))
        if lab:
            items.append((y, ym, 0, 1.0, speech_starts(y), True))   # time-reversed marked clip -> negative
    val = [(f'{th.meta(p)["corpus"]}/{th.meta(p)["stem"]}', th.meta(p)["model"], y) for p, y, _ in va]

    net = Net(torch.load(ROOT / init))
    opt = torch.optim.AdamW([{"params": net.wavlm.parameters(), "lr": 2e-5},
                             {"params": net.head.parameters(), "lr": 2e-4}], weight_decay=1e-2)
    bs = 16; steps = epochs * ((len(items) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [2e-5, 2e-4], total_steps=steps, pct_start=0.1)
    labels = np.array([it[2] for it in items]); pos_w = torch.tensor((labels == 0).sum() / labels.sum())

    sc, auc, tpr = evaluate(net, val)
    print(f"epoch 0 (frozen-feature model): val AUC {auc:.3f} TPR@1/2/5% {tpr[0.01]:.3f}/{tpr[0.02]:.3f}/{tpr[0.05]:.3f}", flush=True)
    best = np.mean(list(tpr.values()))
    for ep in range(1, epochs + 1):
        net.train(); net.wavlm.eval()                              # keep WavLM dropout/LayerDrop off; head dropout on
        order = rng.permutation(len(items)); tot = 0; t0 = time.time()
        for i in range(0, len(order), bs):
            xb, yb, wb = [], [], []
            for j in order[i:i + bs]:
                y, ym, lab, w, starts, rev = items[j]
                r = rng.random(); src = ym if r < 0.2 else (noisy(y) if r < 0.4 else y)
                if rev:
                    src = src[::-1].copy(); s = int(rng.integers(0, max(1, len(src) - SEG + 1)))
                else:
                    s = int(rng.choice(starts)) + int(rng.integers(-4000, 4000))
                    s = min(max(0, s), max(0, len(src) - SEG))
                xb.append(znorm(window(src, s))); yb.append(lab * 0.9 + 0.05); wb.append(w)
            out = net(torch.tensor(np.array(xb)))
            loss = nn.functional.binary_cross_entropy_with_logits(out, torch.tensor(yb), weight=torch.tensor(wb), pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(xb)
        sc, auc, tpr = evaluate(net, val)
        cur = np.mean(list(tpr.values()))
        print(f"epoch {ep} loss {tot / len(items):.3f} ({time.time() - t0:.0f}s): val AUC {auc:.3f} "
              f"TPR@1/2/5% {tpr[0.01]:.3f}/{tpr[0.02]:.3f}/{tpr[0.05]:.3f}{'  *best' if cur > best else ''}", flush=True)
        torch.save(net.state_dict(), ROOT / f"results/finetune{TAG}_last.pt")
        if cur > best:
            best = cur; torch.save(net.state_dict(), ROOT / f"results/finetune{TAG}_best.pt")
            (ROOT / f"results/val_scores_finetune{TAG}.json").write_text(json.dumps(sc))
    print("done", flush=True)


if __name__ == "__main__":
    main()
