"""Small CNN stand-in detector on 1 s log-STFT patches at 16 kHz.

The key test is cross-model generalisation: train on sonic-3.6 (marked) vs sonic-3.5 (unmarked),
then test on sonic-preview (marked) vs sonic-3 (unmarked), models the CNN never saw. Only a
watermark shared by both marked models can carry over; model timbre can't.
"""
import glob, pathlib, sys
import numpy as np, soundfile as sf, torch, torch.nn as nn
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parent.parent
SR, SEG = 16000, 16000
torch.manual_seed(0); np.random.seed(0)


def to16k(x, sr):
    from math import gcd
    g = gcd(sr, SR)
    return signal.resample_poly(x, SR // g, sr // g)


def segments(x):
    out = []
    for i in range(0, len(x) - SEG + 1, SEG // 2):
        s = x[i:i + SEG]
        if np.sqrt(np.mean(s ** 2)) > 0.01:          # skip silence
            out.append(s / (np.sqrt(np.mean(s ** 2)) + 1e-9))
    return out


def load_clips(models, texts):
    clips = []
    for p in sorted(glob.glob(str(ROOT / "audio/corpus2/*.wav"))):
        t, v, m = pathlib.Path(p).stem.split("_", 2)
        if m in models and int(t[1:]) in texts:
            x, sr = sf.read(p)
            clips.append((to16k(x, sr), models[m], m))
    return clips


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.win = torch.hann_window(512)
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 64))
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        S = torch.stft(x, 512, 128, window=self.win, return_complex=True).abs()
        S = torch.log(S + 1e-5)[:, None]
        return self.head(self.f(S).mean((2, 3)))[:, 0]


def train(clips, epochs=12):
    X = torch.tensor(np.array([s for x, y, _ in clips for s in segments(x)]), dtype=torch.float32)
    Y = torch.tensor([y for x, y, _ in clips for s in segments(x)], dtype=torch.float32)
    net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), 32):
            b = perm[i:i + 32]
            xb = X[b] * (1 - 2 * (torch.rand(len(b), 1) < .5))      # random polarity augmentation
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb), Y[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        print(f"  epoch {ep + 1} loss {tot / len(X):.3f}", flush=True)
    return net


@torch.no_grad()
def score(net, x):
    net.eval()
    s = segments(x)
    if not s:
        return float("nan")
    return torch.sigmoid(net(torch.tensor(np.array(s), dtype=torch.float32))).mean().item()


def report(net, clips, label, fn=lambda x: x):
    sc = np.array([score(net, fn(x)) for x, y, _ in clips]); y = np.array([c[1] for c in clips])
    names = sorted(set(c[2] for c in clips))
    per = " ".join(f"{n}={np.mean([s for s, c in zip(sc, clips) if c[2] == n]):.2f}" for n in names)
    print(f"{label:38s} acc={np.mean((sc > .5) == y):.2f}  mean P(marked): {per}", flush=True)


if __name__ == "__main__":
    train_texts, test_texts = range(0, 9), range(9, 12)
    tr = load_clips({"sonic-3.6": 1, "sonic-3.5": 0}, train_texts)
    print(f"training on {len(tr)} clips (sonic-3.6 vs sonic-3.5, texts 0-8)")
    net = train(tr)
    torch.save(net.state_dict(), ROOT / "results/cnn_surrogate.pt")
    same = load_clips({"sonic-3.6": 1, "sonic-3.5": 0}, test_texts)
    cross = load_clips({"sonic-preview": 1, "sonic-3": 0}, range(12))
    report(net, same, "held-out texts, same models")
    report(net, cross, "UNSEEN models (preview vs 3)")
    report(net, same, "held-out, time-reversed", lambda x: x[::-1].copy())
    report(net, cross, "unseen models, time-reversed", lambda x: x[::-1].copy())
