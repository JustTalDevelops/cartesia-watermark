"""Local Cartesia-watermark detector (no API calls).

    .venv/bin/python tools/detect_local.py clip.wav [more.mp3 ...]

Round-4 detector (results/final_detector_r4.json): XLS-R-300M layer-2 frames over 3 s speech windows -> three
conv heads trained with mark-targeted contrastive variants -> mean logit. Threshold set on validation voices.
Sealed4 (48 unseen voices, 14 languages incl. 2 never seen in training, scored once): AUC 0.991,
TPR 94% at FPR 4%. Works on clips from ~3 s; clips with no speech above -50 dBFS get "no speech".
"""
import os, pathlib, subprocess, sys, tempfile
os.environ.setdefault("FEAT", "xlsr2")                 # must be set before importing the feature code
import soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from final_detector import Detector, ROOT


def read_any(path):
    try:
        x, sr = sf.read(path)
    except Exception:                                   # mp3/m4a/etc. via ffmpeg
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", f"{d}/a.wav"], check=True)
            x, sr = sf.read(f"{d}/a.wav")
    return (x if x.ndim == 1 else x.mean(1)), sr


if __name__ == "__main__":
    det = Detector(config=ROOT / "results/final_detector_r4.json")
    for path in sys.argv[1:]:
        x, sr = read_any(path)
        s = det.score(x, sr)
        if s is None:
            print(f"{path}: no speech"); continue
        print(f"{path}: score={s:+.2f} (threshold {det.threshold:+.2f}) -> {'CARTESIA WATERMARK' if s > det.threshold else 'not detected'}")
