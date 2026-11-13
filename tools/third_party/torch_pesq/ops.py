"""Differentiable 16 kHz operations for the optional PESQ training loss.

Static IIR filters are rendered as 4096-sample impulse responses, then applied
by zero-padded FFT convolution. No torchaudio binary extension is required.
"""
import numpy as np
from scipy import signal as scipy_signal
import torch
from torch import nn

_filter_cache = {}


def lfilter(waveform, a_coeffs, b_coeffs, clamp=False):
    a = a_coeffs.detach().cpu().double().numpy()
    b = b_coeffs.detach().cpu().double().numpy()
    n = waveform.shape[-1]
    fft_size = 1 << (n + 4096 - 2).bit_length()
    key = (a.tobytes(), b.tobytes(), fft_size, waveform.dtype, str(waveform.device))
    if key not in _filter_cache:
        impulse = np.zeros(4096)
        impulse[0] = 1
        response = scipy_signal.lfilter(b, a, impulse)
        if not np.isfinite(response).all() or np.max(np.abs(response)) > 1e3:
            raise ValueError("Unstable perceptual-loss filter")
        response = torch.tensor(response, dtype=waveform.dtype, device=waveform.device)
        _filter_cache[key] = torch.fft.rfft(response, n=fft_size)
    out = torch.fft.irfft(torch.fft.rfft(waveform, n=fft_size) * _filter_cache[key], n=fft_size)[..., :n]
    return out.clamp(-1, 1) if clamp else out


class Resample(nn.Module):
    def __init__(self, original_rate, new_rate):
        super().__init__()
        if original_rate != 16000 or new_rate != 16000:
            raise ValueError("This training-loss adapter accepts 16 kHz input only")

    def forward(self, x):
        return x


class Spectrogram(nn.Module):
    def __init__(self, win_length, n_fft, hop_length, window_fn, power, normalized, center):
        super().__init__()
        self.register_buffer("window", window_fn(win_length))
        self.n_fft, self.hop_length, self.power = n_fft, hop_length, power
        self.normalized, self.center = normalized, center

    def forward(self, x):
        z = torch.stft(x, self.n_fft, self.hop_length, window=self.window,
                       center=self.center, normalized=self.normalized, return_complex=True)
        return z.abs().pow(self.power)
