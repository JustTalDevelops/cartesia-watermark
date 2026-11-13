"""Mark-targeted contrastive variants of marked TRAINING clips, as frames of the FEAT set (resumable, sharded).

    [FEAT=xlsr2] python tools/frames_contrast.py <shard> <nshards> <threads>

Each variant keeps the clip's timbre and changes only the 1.5-3.5 kHz band, where the mark lives. Labels follow
Cartesia's own detector on the same edits (docs/findings.md section 7):
  delay10  : 1.5-3.5 kHz band delayed 10 ms                    -> still marked  (1/5/20 ms all detected)
  delay40  : 1.5-3.5 kHz band delayed 40 ms                    -> not marked    (33-100 ms: 5/5 not detected)
  twinband : 1.5-3.5 kHz band replaced by the sonic-3.5 twin's -> not marked    (3/3 not detected)
             (same voice and text rendered by an unmarked model)
Validation voices are skipped, so validation stays natural audio only.
"""
import glob, pathlib, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train_head as th
from frames import frames, to16k, OUT
from probe import butter

ROOT = th.ROOT


def band(x, sr):
    return butter(x, sr, "band", [1500, 3500])


def delay_band(x, sr, ms):
    b = band(x, sr); d = int(round(ms * sr / 1000))
    return x - b + np.concatenate([np.zeros(d), b[:len(b) - d]])


def twin_band(x, twin, sr):
    t = np.zeros_like(x); n = min(len(x), len(twin)); t[:n] = twin[:n]
    return x - band(x, sr) + band(t, sr)


if __name__ == "__main__":
    shard, nshards = int(sys.argv[1]), int(sys.argv[2])
    torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    paths = []
    for corpus in th.TRAIN_CORPORA:
        for p in sorted(glob.glob(str(ROOT / f"audio/{corpus}/*.wav"))):
            m = th.meta(p)
            if m["model"] in th.MARKED and not th.is_val(m):
                paths.append(p)
    paths = paths[shard::nshards]
    for i, p in enumerate(paths):
        p = pathlib.Path(p); m = th.meta(p); x = None
        twin_path = p.with_name(p.name.replace(m["model"], "sonic-3.5"))
        for variant in ("delay10", "delay40", "twinband"):
            out = OUT / variant / m["corpus"] / f"{m['stem']}.npz"
            if out.exists() or (variant == "twinband" and not twin_path.exists()):
                continue
            if x is None:
                x, sr = sf.read(p)
            if variant == "delay10":
                z = delay_band(x, sr, 10)
            elif variant == "delay40":
                z = delay_band(x, sr, 40)
            else:
                tw, tsr = sf.read(twin_path); assert tsr == sr
                z = twin_band(x, tw, sr)
            h, rms = frames(to16k(z, sr))
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, h=h, rms=rms)
        if i % 100 == 0:
            print(f"shard {shard}: {i}/{len(paths)}", flush=True)
    print(f"shard {shard} done", flush=True)
