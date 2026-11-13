"""The 121 probe items Cartesia's detector labelled (used by train_head.py to report agreement with Cartesia).

Each item rebuilds the exact audio we sent (same transforms, same seeds) and carries Cartesia's answer.
Needs audio/corpus (tools/corpus.py), audio/corpus3 and audio/fmt (format probes from the first session).
`evaluate()` / the CLI score the early raw-waveform CNN detectors in experiments/.
Items where Cartesia flagged unmarked (sonic-3.5 / synthetic) audio are known false positives; they're
reported separately because a good watermark detector *should* disagree with Cartesia there.
"""
import glob, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from probe import load, butter, resample, noise, ffmpeg_roundtrip
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parent.parent


def stft_edit(x, sr, lo, hi, mode, seed=0):
    f, t, Z = signal.stft(x, sr, nperseg=1024)
    b = (f >= lo) & (f < hi)
    rng = np.random.default_rng(seed)
    if mode == "randphase":
        Z[b] = np.abs(Z[b]) * np.exp(1j * rng.uniform(0, 2 * np.pi, Z[b].shape))
    elif mode == "smoothmag":
        from scipy.ndimage import uniform_filter1d
        Z[b] = uniform_filter1d(np.abs(Z[b]), 9, axis=1) * np.exp(1j * np.angle(Z[b]))
    return signal.istft(Z, sr, nperseg=1024)[1][:len(x)]


def items():
    """Yield (name, audio, sr, cartesia_label, kind) with kind in {marked, unmarked, fp}."""
    P, sr = load("audio/corpus/t0_daniel_sonic-3.6.wav"); N, _ = load("audio/corpus/t0_daniel_sonic-3.5.wav")
    T = {
        "orig": (lambda x: x, 1, 0), "gain-20dB": (lambda x: x * .1, 1, 0), "invert": (lambda x: -x, 1, 0),
        "reverse": (lambda x: x[::-1].copy(), 0, 0), "first5s": (lambda x: x[:5 * sr], 1, 0),
        "last6s": (lambda x: x[-6 * sr:], 1, 1), "shift1": (lambda x: x[1:], 1, 0), "shift257": (lambda x: x[257:], 1, 0),
        "lp8k": (lambda x: butter(x, sr, "low", 8000), 1, 1), "lp4k": (lambda x: butter(x, sr, "low", 4000), 1, 0),
        "lp2k": (lambda x: butter(x, sr, "low", 2000), 0, 0), "lp1k": (lambda x: butter(x, sr, "low", 1000), 0, 0),
        "hp1k": (lambda x: butter(x, sr, "high", 1000), 1, 0), "hp4k": (lambda x: butter(x, sr, "high", 4000), 0, 0),
        "hp8k": (lambda x: butter(x, sr, "high", 8000), 0, 0),
        "rs16k": (lambda x: resample(resample(x, sr, 16000), 16000, sr), 1, 0),
        "rs8k": (lambda x: resample(resample(x, sr, 8000), 8000, sr), 1, 1),
        "mp3_64k": (lambda x: ffmpeg_roundtrip(x, sr, ["-b:a", "64k"], "mp3"), 1, 0),
        "mp3_32k": (lambda x: ffmpeg_roundtrip(x, sr, ["-b:a", "32k"], "mp3"), 1, 0),
        "opus_16k": (lambda x: ffmpeg_roundtrip(x, sr, ["-c:a", "libopus", "-b:a", "16k"], "ogg"), 1, 0),
        "snr20": (lambda x: noise(x, 20), 1, 0), "snr10": (lambda x: noise(x, 10), 0, 0), "snr0": (lambda x: noise(x, 0), 0, 0),
        "speed1.05": (lambda x: signal.resample(x, int(len(x) / 1.05)), 1, 0),
    }
    for k, (fn, lp, ln) in T.items():
        yield f"A36 {k}", fn(P), sr, lp, "marked" if lp else "unmarked"
        yield f"A35 {k}", fn(N), sr, ln, "fp" if ln else "unmarked"
    for f in sorted(glob.glob("audio/fmt/*")):
        x, s = load(f)
        is35 = "3.5" in f
        yield f"fmt {pathlib.Path(f).name}", x, s, 1, "fp" if is35 else "marked"
    for f in sorted(glob.glob("audio/corpus/*sonic-3.5.wav")):
        yield f"neg {pathlib.Path(f).stem}", load(f)[0], sr, 0, "unmarked"
    rng = np.random.default_rng(1)
    yield "zeros6s", np.zeros(6 * sr), sr, 0, "unmarked"
    yield "white", .05 * rng.standard_normal(8 * sr), sr, 0, "unmarked"
    yield "sine1k", .2 * np.sin(2 * np.pi * 1000 * np.arange(8 * sr) / sr), sr, 1, "fp"
    yield "lpnoise", butter(.1 * rng.standard_normal(8 * sr), sr, "low", 3000, 2), sr, 0, "unmarked"
    clips = {"A": "audio/corpus/t0_daniel_sonic-3.6.wav", "B": "audio/corpus/t1_jacqueline_sonic-3.6.wav",
             "C": "audio/corpus/t3_jacqueline_sonic-3.6.wav"}
    n = min(len(P), len(N)); Pn, Nn = P[:n], N[:n]
    band = lambda x, lo, hi: butter(x, sr, "band", [lo, hi])
    for lo, hi, only in [(2000, 4000, 0), (1000, 4000, 1), (1500, 3500, 1)]:
        yield f"A only 3.6 {lo}-{hi}", band(Pn, lo, hi), sr, only, "marked" if only else "unmarked"
        yield f"A 3.5+3.6[{lo}-{hi}]", Nn - band(Nn, lo, hi) + band(Pn, lo, hi), sr, 0, "unmarked"
        yield f"A 3.6 w/ 3.5[{lo}-{hi}]", Pn - band(Pn, lo, hi) + band(Nn, lo, hi), sr, 0, "unmarked"
    for frac in [.75, .5, .25]:
        k = int(frac * n); yield f"A splice {frac}", np.concatenate([Pn[:k], Nn[k:]]), sr, 1, "marked"
    for s0 in np.arange(0, len(P) / sr - 5, 2.5):
        yield f"A win {s0}", P[int(s0 * sr):int((s0 + 5) * sr)], sr, 1, "marked"
    for k, lab in [("A", (1, 1, 0)), ("B", (0, 1, 0))]:
        x, _ = load(clips[k])
        for (mode, lo, hi), l in zip([("randphase", 1500, 3500), ("smoothmag", 1500, 3500), ("randphase", 0, 22050)], lab):
            yield f"{k} {mode} {lo}-{hi}", stft_edit(x, sr, lo, hi, mode), sr, l, "marked" if l else "unmarked"
    for lo in range(1000, 4500, 250):
        yield f"A notch {lo}", P - butter(P, sr, "band", [lo, lo + 250], order=6), sr, 1, "marked"
    for k, l in [(.5, 0), (1, 0), (2, 1)]:
        m = int(k * sr); yield f"A first {k}s", np.concatenate([P[:m], N[m:]]), sr, l, "marked" if l else "unmarked"
    for snr, l in [(12, 0), (14, 0), (16, 1), (18, 1)]:
        yield f"A snr {snr}", noise(P, snr), sr, l, "marked" if l else "unmarked"
    for k in ["B", "C"]:
        x, _ = load(clips[k]); y, _ = load(clips[k].replace("3.6", "3.5")); m = min(len(x), len(y))
        yield f"{k} reversed", x[::-1].copy(), sr, 0, "unmarked"
        yield f"{k} 3.6 w/ 3.5[1500-3500]", x[:m] - band(x[:m], 1500, 3500) + band(y[:m], 1500, 3500), sr, 0, "unmarked"
        yield f"{k} only 3.6 1500-3500", band(x[:m], 1500, 3500), sr, 1, "marked"


def _td():
    sys.path.insert(0, str(ROOT / "experiments"))
    import train_detector
    return train_detector


def evaluate(net, verbose=False, pool="mean"):
    td = _td()
    rows = []
    for name, x, sr, lab, kind in items():
        y = td.prep(np.asarray(x, dtype=np.float64), sr)
        s = td.segments(y, hop=td.SEG // 4)
        if not s:
            p = 0.0
        else:
            with torch.no_grad():
                net.eval(); probs = torch.sigmoid(net(torch.tensor(np.array(s)))).numpy()
            p = float(probs.mean() if pool == "mean" else probs.max())
        rows.append((name, kind, lab, p))
        if verbose:
            print(f"{name:32s} {kind:9s} cartesia={lab} local={p:.2f} {'OK' if (p > .5) == lab else 'XX'}")
    for kind in ["marked", "unmarked", "fp"]:
        r = [x for x in rows if x[1] == kind]
        agree = np.mean([(x[3] > .5) == x[2] for x in r])
        print(f"  agreement with Cartesia on {kind:9s} items ({len(r):2d}): {agree:.2f}")
    core = [x for x in rows if x[1] != "fp"]
    print(f"  OVERALL agreement (excluding Cartesia's false positives): {np.mean([(x[3] > .5) == x[2] for x in core]):.3f}")
    return rows


if __name__ == "__main__":
    if "--v3" in sys.argv:
        _td(); import train_v3
        net = train_v3.WaveNet()
    else:
        net = _td().Net()
    net.load_state_dict(torch.load(sys.argv[1]))
    evaluate(net, verbose="-v" in sys.argv, pool="max" if "--max" in sys.argv else "mean")
