"""Local Cartesia-watermark detector v3: raw-waveform 1D CNN, distilled from Cartesia's labels.

Training data
  weak   : natural corpus3 clips, labelled by model (marked = sonic-3.6 / sonic-preview)
  strong : Cartesia-labelled transforms from results/distill.json (same timbre, different mark state), weight 3
  shortcut killers: white/pink noise and noisy unmarked speech (negatives), reversed marked speech (negative)
Voices 1 and 5 (in the validation set) and 24-29 (held out) are never trained on.
"""
import json, pathlib, sys
import numpy as np, soundfile as sf, torch, torch.nn as nn
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
import train_detector as td                       # prep(): 16 kHz + 1-4.5 kHz band; segments()

ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(20)
SEG = td.SEG                                       # 2 s at 16 kHz
HELD_OUT = set(range(24, 30)) | {1, 5}


class WaveNet(nn.Module):
    """Strided conv encoder (hop 320 = 20 ms) -> per-frame logits -> attention pooling."""
    def __init__(self, ch=(32, 64, 128, 192), strides=(4, 4, 4, 5)):
        super().__init__()
        layers, c_in = [nn.Conv1d(1, ch[0], 7, padding=3), nn.GELU()], ch[0]
        for c_out, s in zip(ch, strides):
            layers += [nn.Conv1d(c_in, c_out, 2 * s + 1, stride=s, padding=s), nn.BatchNorm1d(c_out), nn.GELU(),
                       nn.Conv1d(c_out, c_out, 3, padding=1), nn.BatchNorm1d(c_out), nn.GELU()]
            c_in = c_out
        self.enc = nn.Sequential(*layers)
        self.ctx = nn.GRU(c_in, 96, batch_first=True, bidirectional=False)   # unidirectional: time direction matters
        self.frame = nn.Linear(96, 1); self.att = nn.Linear(96, 1)

    def frames(self, x):
        h, _ = self.ctx(self.enc(x[:, None]).transpose(1, 2))
        return self.frame(h)[..., 0], self.att(h)[..., 0]

    def forward(self, x):
        f, a = self.frames(x)
        w = torch.softmax(a, 1)
        return (f * w).sum(1)


def natural_items():
    out = []
    for p in sorted((ROOT / "audio/corpus3").glob("*.wav")):
        t, v, m = p.stem.split("_", 2)
        if int(v[1:]) in HELD_OUT:
            continue
        x, sr = sf.read(p)
        out.append((td.prep(x, sr), td.MARKED[m], 1.0, m))
    return out


def distilled_items():
    d = json.loads((ROOT / "results/distill.json").read_text()) if (ROOT / "results/distill.json").exists() else {}
    out = []
    for k, v in d.items():
        if v["stage"] != "transform" or v["voice"] in HELD_OUT:
            continue
        x, sr = sf.read(ROOT / v["path"])
        out.append((td.prep(x, sr), int(v["detected"]), 3.0, "distill"))
    return out


def shortcut_negatives(n=150, seed=0):
    rng = np.random.default_rng(seed); out = []
    nat = [it for it in natural_items() if it[1] == 0]
    for i in range(n):
        kind = i % 3
        if kind == 0:
            x = rng.standard_normal(SEG * 2)
        elif kind == 1:
            x = np.cumsum(rng.standard_normal(SEG * 2)); x -= np.convolve(x, np.ones(400) / 400, "same")
        else:                                      # unmarked speech + noise at 5-25 dB SNR
            y = nat[rng.integers(len(nat))][0]
            x = y + rng.standard_normal(len(y)) * y.std() * 10 ** (-rng.uniform(5, 25) / 20)
        out.append((td.prep(x, 16000), 0, 1.0, "noise"))
    return out


def build(items):
    X, Y, W = [], [], []
    for y, lab, w, src in items:
        for s in td.segments(y):
            X.append(s); Y.append(lab); W.append(w)
            if lab == 1 and src != "distill":
                X.append(s[::-1].copy()); Y.append(0); W.append(1.0)
    return torch.tensor(np.array(X)), torch.tensor(Y, dtype=torch.float32), torch.tensor(W, dtype=torch.float32)


def train(items, epochs=10, log=print):
    X, Y, Wt = build(items)
    log(f"  {len(X)} segments, {int(Y.sum())} positive")
    net = WaveNet(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    steps = epochs * ((len(X) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            l = nn.functional.binary_cross_entropy_with_logits(net(td.augment(X[b])), Y[b], weight=Wt[b])
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += l.item() * len(b)
        log(f"  epoch {ep + 1:2d} loss {tot / len(X):.3f}")
    return net


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    items = natural_items() + distilled_items() + shortcut_negatives()
    print(f"{len(items)} training items ({sum(1 for i in items if i[3] == 'distill')} Cartesia-labelled)", flush=True)
    net = train(items, epochs, log=lambda s: print(s, flush=True))
    torch.save(net.state_dict(), ROOT / "results/detector_v3.pt")
    print("saved results/detector_v3.pt", flush=True)
