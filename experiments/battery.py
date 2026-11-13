"""Robustness battery: apply ~30 edits to a marked clip and its unmarked twin, ask Cartesia's detector.

    python experiments/battery.py marked.wav unmarked_twin.wav
"""
import sys, pathlib
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
from probe import *

pos, sr = load(sys.argv[1]); neg, _ = load(sys.argv[2])
n = min(len(pos), len(neg))
T = {
 "orig": lambda x: x,
 "gain-20dB": lambda x: x * 0.1,
 "invert": lambda x: -x,
 "reverse": lambda x: x[::-1],
 "first5s": lambda x: x[:5*sr],
 "first4s": lambda x: x[:4*sr],
 "last6s": lambda x: x[-6*sr:],
 "shift1": lambda x: x[1:],
 "shift257": lambda x: x[257:],
 "lp8k": lambda x: butter(x, sr, "low", 8000),
 "lp4k": lambda x: butter(x, sr, "low", 4000),
 "lp2k": lambda x: butter(x, sr, "low", 2000),
 "lp1k": lambda x: butter(x, sr, "low", 1000),
 "hp1k": lambda x: butter(x, sr, "high", 1000),
 "hp4k": lambda x: butter(x, sr, "high", 4000),
 "hp8k": lambda x: butter(x, sr, "high", 8000),
 "rs16k": lambda x: resample(resample(x, sr, 16000), 16000, sr),
 "rs8k": lambda x: resample(resample(x, sr, 8000), 8000, sr),
 "mp3_64k": lambda x: ffmpeg_roundtrip(x, sr, ["-b:a", "64k"], "mp3"),
 "mp3_32k": lambda x: ffmpeg_roundtrip(x, sr, ["-b:a", "32k"], "mp3"),
 "opus_16k": lambda x: ffmpeg_roundtrip(x, sr, ["-c:a", "libopus", "-b:a", "16k"], "ogg"),
 "snr20": lambda x: noise(x, 20),
 "snr10": lambda x: noise(x, 10),
 "snr0": lambda x: noise(x, 0),
 "speed1.05": lambda x: signal.resample(x, int(len(x) / 1.05)),
 "speed0.95": lambda x: signal.resample(x, int(len(x) / 0.95)),
}
only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(T)
for k in only:
    try:
        print(f"{k:12s} pos={detect_array(T[k](pos), sr, k)!s:6s} neg={detect_array(T[k](neg), sr, k)!s}", flush=True)
    except Exception as e:
        print(k, "ERR", e)
