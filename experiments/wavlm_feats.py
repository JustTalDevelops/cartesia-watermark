"""Extract WavLM-base-plus layer features (mean+std pooled) for every natural clip."""
import glob, pathlib, sys
import numpy as np, soundfile as sf, torch
from scipy import signal
from transformers import WavLMModel
ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.set_num_threads(8)
band = "--band" in sys.argv                         # optionally restrict input to 1-4.5 kHz
m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
sos = signal.butter(6, [1000, 4500], btype="band", fs=16000, output="sos")
keys, feats = [], []
paths = sorted(glob.glob(str(ROOT / "audio/corpus3/*.wav"))) + sorted(glob.glob(str(ROOT / "audio/corpus4/*.wav")))
for i, p in enumerate(paths):
    x, sr = sf.read(p)
    y = signal.resample_poly(x, 160, 441).astype(np.float32)
    if band:
        y = signal.sosfiltfilt(sos, y).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-9)
    with torch.no_grad():
        hs = m(torch.tensor(y)[None], output_hidden_states=True).hidden_states
    feats.append(np.stack([np.concatenate([h[0].mean(0).numpy(), h[0].std(0).numpy()]) for h in hs]))
    keys.append(str(pathlib.Path(p).relative_to(ROOT)))
    if i % 200 == 0:
        print(i, "/", len(paths), flush=True)
np.savez(ROOT / f"results/wavlm_feats{'_band' if band else ''}.npz", keys=np.array(keys), feats=np.array(feats, dtype=np.float32))
print("done", flush=True)
