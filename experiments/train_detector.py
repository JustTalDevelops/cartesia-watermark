"""Local Cartesia-watermark detector v2.

Lessons from the black-box probes baked in:
  * the mark lives in ~1.5-3.5 kHz and survives 16 kHz resampling  -> 16 kHz input, band 1-4.5 kHz
  * phase matters (random phase kills it), magnitude blur doesn't  -> complex STFT (re, im) input
  * polarity-invariant, gain-invariant, robust to mild noise        -> augmentations
  * time reversal kills it                                          -> reversed marked audio = negative
Positives: sonic-3.6 (fluent-bird) + sonic-preview (wise-puddle). Negatives: sonic-3.5 + sonic-3.
Validation: unseen voices, and leave-model-out (train 3.6/3.5, test preview/3).
"""
import glob, json, pathlib, sys
import numpy as np, soundfile as sf, torch, torch.nn as nn
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parent.parent
SR, SEG, NFFT, HOP = 16000, 32000, 512, 128
F_LO, F_HI = 1000, 4500
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(10)
MARKED = {"sonic-3.6": 1, "sonic-preview": 1, "sonic-3.5": 0, "sonic-3": 0}
_band = signal.butter(6, [F_LO, F_HI], btype="band", fs=SR, output="sos")


def to16k(x, sr):
    from math import gcd
    g = gcd(sr, SR)
    return signal.resample_poly(x, SR // g, sr // g)


def prep(x, sr):
    """Resample to 16 kHz and band-limit; shared by training and inference."""
    return signal.sosfiltfilt(_band, to16k(x, sr)).astype(np.float32)


def segments(y, hop=SEG // 2):
    out = []
    for i in range(0, max(1, len(y) - SEG + 1), hop):
        s = y[i:i + SEG]
        if len(s) < SEG:
            s = np.pad(s, (0, SEG - len(s)))
        r = np.sqrt(np.mean(s ** 2))
        if r > 1e-4:
            out.append(s / r)
    return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("win", torch.hann_window(NFFT))
        f = np.fft.rfftfreq(NFFT, 1 / SR)
        self.lo, self.hi = int(np.argmax(f >= F_LO)), int(np.argmax(f >= F_HI))
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 96), blk(96, 128))
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        Z = torch.stft(x, NFFT, HOP, window=self.win, return_complex=True)[:, self.lo:self.hi]
        mag = Z.abs() + 1e-6
        # complex STFT normalised per bin by magnitude scale -> phase information kept, loudness removed
        feat = torch.stack([Z.real / mag.mean(-1, keepdim=True), Z.imag / mag.mean(-1, keepdim=True), torch.log(mag)], 1)
        return self.head(self.f(feat).mean((2, 3)))[:, 0]


def load_corpus(pattern):
    items = []
    for p in sorted(glob.glob(str(ROOT / pattern))):
        stem = pathlib.Path(p).stem
        t, v, m = stem.split("_", 2)
        if m not in MARKED:
            continue
        x, sr = sf.read(p)
        items.append(dict(y=prep(x, sr), label=MARKED[m], model=m, voice=v, text=t, path=p))
    return items


def build(items, hard_negatives=True):
    X, Y = [], []
    for it in items:
        for s in segments(it["y"]):
            X.append(s); Y.append(it["label"])
            if hard_negatives and it["label"] == 1:
                X.append(s[::-1].copy()); Y.append(0)          # reversed marked speech: Cartesia says unmarked
    return torch.tensor(np.array(X)), torch.tensor(Y, dtype=torch.float32)


def augment(xb):
    b = len(xb)
    xb = xb * torch.where(torch.rand(b, 1) < .5, -1.0, 1.0)                        # polarity
    xb = xb * torch.exp(torch.randn(b, 1) * 0.3)                                    # gain
    snr = 25 + 20 * torch.rand(b, 1)                                                # 25-45 dB noise
    xb = xb + torch.randn_like(xb) * xb.std(1, keepdim=True) * 10 ** (-snr / 20)
    shift = int(torch.randint(0, HOP, (1,)))
    return torch.roll(xb, shift, 1)


def train(items, epochs=15, log=print):
    X, Y = build(items)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(X) + 63) // 64))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            loss = nn.functional.binary_cross_entropy_with_logits(net(augment(X[b])), Y[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        log(f"  epoch {ep + 1:2d} loss {tot / len(X):.3f}")
    return net


@torch.no_grad()
def score_array(net, y):
    """y: already prep()'d 16 kHz band-limited audio. Returns mean P(marked) over 2 s windows."""
    net.eval()
    s = segments(y)
    if not s:
        return float("nan")
    return torch.sigmoid(net(torch.tensor(np.array(s)))).mean().item()


def score_file(net, path):
    x, sr = sf.read(path)
    return score_array(net, prep(x if x.ndim == 1 else x.mean(1), sr))


def report(net, items, label, fn=None):
    sc = np.array([score_array(net, fn(it["y"]) if fn else it["y"]) for it in items])
    y = np.array([it["label"] for it in items])
    per = " ".join(f"{m}={np.mean(sc[[it['model'] == m for it in items]]):.2f}" for m in MARKED if any(it["model"] == m for it in items))
    acc = np.mean((sc > .5) == y)
    print(f"{label:40s} acc={acc:.3f}  mean P(marked): {per}", flush=True)
    return acc


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    items = load_corpus("audio/corpus3/*.wav")
    print(len(items), "clips loaded", flush=True)
    vid = lambda it: int(it["voice"][1:])
    if which in ("1", "both"):
        # Leave-model-out: the real test of 'is it the watermark or the timbre?'
        tr = [it for it in items if it["model"] in ("sonic-3.6", "sonic-3.5") and vid(it) < 24]
        te = [it for it in items if it["model"] in ("sonic-preview", "sonic-3") and vid(it) >= 24]
        print(f"\n[1] train 3.6 vs 3.5 (voices 0-23, {len(tr)} clips) -> test preview vs 3 on unseen voices", flush=True)
        net = train(tr, epochs)
        report(net, te, "UNSEEN models + voices")
        report(net, te, "  same, time-reversed", lambda y: y[::-1].copy())
        torch.save(net.state_dict(), ROOT / "results/detector_v2_lmo.pt")
    if which in ("2", "both"):
        # All four models, unseen voices
        tr = [it for it in items if vid(it) < 24]; te = [it for it in items if vid(it) >= 24]
        print(f"\n[2] train all 4 models (voices 0-23, {len(tr)} clips) -> test unseen voices", flush=True)
        net = train(tr, epochs)
        report(net, te, "unseen voices")
        report(net, te, "  same, time-reversed", lambda y: y[::-1].copy())
        torch.save(net.state_dict(), ROOT / "results/detector_v2.pt")
