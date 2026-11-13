"""Local Cartesia-watermark detector v4: trained on rule-labelled transforms of marked/unmarked speech.

Every rule below was measured against Cartesia's detector (see docs/findings.md / results/*.json) and checked on
new clips (13/14 agreement). Because the same transforms appear in both classes and marked-timbre audio
appears in both classes, neither model timbre nor processing artefacts predict the label; only the mark does.
Cartesia's distillation labels (results/distill.json) are NOT used for training; they are a held-out test.
"""
import glob, pathlib, random, sys
import numpy as np, soundfile as sf, torch, torch.nn as nn
from scipy import signal
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
import train_detector as td
from train_v3 import WaveNet

ROOT = pathlib.Path(__file__).resolve().parent.parent
SR = 44100; CROP = int(2.2 * SR)
HELD_OUT = set(range(24, 30)) | {1, 5}
MARKED = {"sonic-3.6", "sonic-preview"}
TWIN = {"sonic-3.6": "sonic-3.5", "sonic-preview": "sonic-3", "sonic-3.5": "sonic-3.6", "sonic-3": "sonic-preview"}


def bf(x, kind, fc, order=6):
    return signal.sosfiltfilt(signal.butter(order, fc, btype=kind, fs=SR, output="sos"), x)


def band(x, lo=1500, hi=3500):
    return bf(x, "band", [lo, hi])


def shift(x, n):
    return np.concatenate([np.zeros(n), x[:len(x) - n]])


def noisy(x, snr, r):
    return x + r.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))


def randphase(x, r):
    f, t, Z = signal.stft(x, SR, nperseg=1024)
    Z = np.abs(Z) * np.exp(1j * r.uniform(0, 2 * np.pi, Z.shape))
    return signal.istft(Z, SR, nperseg=1024)[1][:len(x)]


# (name, fn(m, u, m2, r) -> audio, label-if-host-marked). Host unmarked => label 0 always.
KEEP = [
    ("identity", lambda m, u, m2, r: m),
    ("notch", lambda m, u, m2, r: (lambda lo: m - bf(m, "band", [lo, lo + 250]))(r.uniform(1000, 4250))),
    ("noise_hi", lambda m, u, m2, r: noisy(m, r.uniform(18, 35), r)),
    ("bandshift_small", lambda m, u, m2, r: m - band(m) + shift(band(m), int(r.uniform(0, .015) * SR))),
    ("band_only", lambda m, u, m2, r: band(m)),
    ("lowpass_hi", lambda m, u, m2, r: bf(m, "low", r.uniform(4200, 8000))),
    ("highpass_lo", lambda m, u, m2, r: bf(m, "high", r.uniform(200, 1000))),
    ("marked_band_swap", lambda m, u, m2, r: m - band(m) + band(m2)),
]
KILL = [
    ("reverse", lambda m, u, m2, r: m[::-1].copy()),
    ("bandshift_big", lambda m, u, m2, r: m - band(m) + shift(band(m), int(r.uniform(.04, .2) * SR))),
    ("unmarked_band_swap", lambda m, u, m2, r: m - band(m) + band(u)),
    ("noise_lo", lambda m, u, m2, r: noisy(m, r.uniform(0, 12), r)),
    ("lowpass_lo", lambda m, u, m2, r: bf(m, "low", r.uniform(1000, 2800))),
    ("highpass_hi", lambda m, u, m2, r: bf(m, "high", r.uniform(1700, 4000))),
    ("randphase", lambda m, u, m2, r: randphase(m, r)),
    ("mix_minority", lambda m, u, m2, r: (lambda a: a * m + (1 - a) * u)(r.uniform(.2, .65))),
]


class Data(torch.utils.data.Dataset):
    def __init__(self, n, seed):
        self.clips = {}
        for p in sorted(glob.glob(str(ROOT / "audio/corpus3/*.wav"))):
            t, v, mdl = pathlib.Path(p).stem.split("_", 2)
            if int(v[1:]) in HELD_OUT:
                continue
            x, _ = sf.read(p); self.clips[(t, v, mdl)] = x
        self.keys = list(self.clips); self.n = n; self.seed = seed

    def crop(self, x, r):
        if len(x) <= CROP:
            return np.pad(x, (0, CROP - len(x)))
        i = r.integers(0, len(x) - CROP); return x[i:i + CROP]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = np.random.default_rng((self.seed, i))
        t, v, mdl = self.keys[r.integers(len(self.keys))]
        host = self.crop(self.clips[(t, v, mdl)], r)
        twin = self.crop(self.clips[(t, v, TWIN[mdl])], r)
        others = [k for k in self.keys if k[2] in MARKED and k[1] != v]
        m2 = self.crop(self.clips[others[r.integers(len(others))]], r)
        marked = mdl in MARKED
        if r.random() < .5:
            name, fn = KEEP[r.integers(len(KEEP))]; lab = int(marked)
        else:
            name, fn = KILL[r.integers(len(KILL))]; lab = 0
        y = td.prep(fn(host, twin, m2, r), SR)[:td.SEG]
        y = np.pad(y, (0, td.SEG - len(y)))
        y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9)
        return torch.tensor(y, dtype=torch.float32), torch.tensor(float(lab))


def train(epochs=12, per_epoch=6000, log=print):
    net = WaveNet(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (per_epoch // 64 + 1))
    for ep in range(epochs):
        dl = torch.utils.data.DataLoader(Data(per_epoch, ep), batch_size=64, num_workers=10, shuffle=False)
        net.train(); tot = n = 0
        for xb, yb in dl:
            l = nn.functional.binary_cross_entropy_with_logits(net(td.augment(xb)), yb)
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += l.item() * len(xb); n += len(xb)
        log(f"  epoch {ep + 1:2d} loss {tot / n:.3f}")
        torch.save(net.state_dict(), ROOT / "results/detector_v4.pt")
    return net


if __name__ == "__main__":
    torch.manual_seed(0); torch.set_num_threads(10)
    train(int(sys.argv[1]) if len(sys.argv) > 1 else 12, log=lambda s: print(s, flush=True))
    print("saved results/detector_v4.pt", flush=True)
