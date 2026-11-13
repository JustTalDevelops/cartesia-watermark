"""Reconstruct a native high band without copying its original fine structure."""
import numpy as np
import torch
from scipy.fft import rfft, irfft, rfftfreq

from remover import resample


@torch.inference_mode()
def predict_absolute(model, audio, sample_rate, chunk_seconds=6, model_sample_rate=16000):
    """Resample a rebuilder's absolute output, not its input-relative residual."""
    x = np.asarray(audio, np.float64)
    if x.ndim not in (1, 2) or not x.size or not np.isfinite(x).all() or chunk_seconds <= 1 or model_sample_rate < 8000:
        raise ValueError('Expected finite nonempty audio and chunks longer than one second')
    mono = x.ndim == 1
    channels = x[:, None] if mono else x
    model_audio = resample(channels, sample_rate, model_sample_rate)
    size, overlap = round(chunk_seconds * model_sample_rate), model_sample_rate
    accum, weight_sum = np.zeros_like(model_audio), np.zeros(len(model_audio))
    device = next(model.parameters()).device
    for start in range(0, len(model_audio), size-overlap):
        end = min(len(model_audio), start+size)
        tensor = torch.tensor(model_audio[start:end].T.copy(), dtype=torch.float32, device=device)
        prediction = model(tensor).cpu().numpy().T
        if prediction.shape != tensor.T.shape:
            raise ValueError('Reconstructor must preserve model-rate sample and channel counts')
        weight = np.ones(end-start)
        if start:
            fade = min(overlap, end-start)
            weight[:fade] = np.linspace(0, 1, fade+2)[1:-1]
        if end < len(model_audio):
            fade = min(overlap, end-start)
            weight[-fade:] = np.linspace(1, 0, fade+2)[1:-1]
        accum[start:end] += prediction * weight[:, None]
        weight_sum[start:end] += weight
        if end == len(model_audio):
            break
    predicted = resample(accum / weight_sum[:, None], model_sample_rate, sample_rate)[:len(x)]
    return predicted[:, 0] if mono else predicted


def replace_native_band(original, prediction, sample_rate, low=6000, high=8000,
                        match_power=True, peak_limit=.999):
    """Only one scalar of original band power is retained, per channel.

    Original target-band samples, phases, and time-varying magnitudes are not
    used in the replacement. Remaining native frequencies keep their spectrum.
    """
    x, y = np.asarray(original, np.float64), np.asarray(prediction, np.float64)
    if x.shape != y.shape or not x.size:
        raise ValueError('Expected matching nonempty audio')
    if not 0 < low < high <= sample_rate/2:
        raise ValueError('Invalid native replacement band')
    mono = x.ndim == 1
    x, y = (x[:, None], y[:, None]) if mono else (x, y)
    size = len(x)
    reference, rebuilt = rfft(x, axis=0), rfft(y, axis=0)
    frequency = rfftfreq(size, 1/sample_rate)
    band = (frequency >= low) & (frequency <= high)
    scale = np.ones(x.shape[1])
    if match_power:
        a, b = ((np.abs(s[band])**2).sum(0) for s in (reference, rebuilt))
        if np.any((b <= 1e-24) & (a > 1e-24)):
            raise ValueError('Reconstructor predicted no target-band energy')
        scale = np.sqrt(a / np.maximum(b, 1e-24))
    reference[band] = rebuilt[band] * scale
    out = irfft(reference, n=size, axis=0)
    gain = 1.0
    peak = float(np.abs(out).max())
    if peak_limit is not None and peak > peak_limit:
        gain = peak_limit/peak
        out *= gain
    return (out[:, 0] if mono else out), {'band_power_scale': scale.tolist(), 'uniform_gain': gain}
