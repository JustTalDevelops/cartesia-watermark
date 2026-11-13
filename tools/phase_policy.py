"""Native-rate analytic-band phase policies for quality-constrained research.

This helper never filters the complete output. It rotates analytic carriers
inside selected bands and optionally applies one uniform gain to avoid clipping.
Policy parameters can be fitted against actual responses after fidelity checks.
"""
import numpy as np
from scipy.fft import fft, ifft, fftfreq, next_fast_len


def apply_band_phase(audio, sample_rate, bands=1, low_hz=1500, high_hz=3500,
                     depth_radians=1.5, rate_hz=4.0, seed=2028, peak_limit=.999):
    x = np.asarray(audio, np.float64)
    if x.ndim not in (1, 2) or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite mono or multichannel audio')
    if bands < 1 or not 0 < low_hz < high_hz < sample_rate / 2:
        raise ValueError('Invalid phase bands')
    if not 0 <= depth_radians <= np.pi or not 0 < rate_hz <= 40:
        raise ValueError('Invalid modulation range')
    if peak_limit is not None and not 0 < peak_limit <= 1:
        raise ValueError('Peak limit must be in (0, 1]')
    mono = x.ndim == 1
    channels = x[:, None] if mono else x
    pad = round(sample_rate * .1)
    size = next_fast_len(len(x) + 2 * pad)
    padded = np.pad(channels, ((pad, size-len(x)-pad), (0, 0)))
    spectrum = fft(padded, axis=0)
    frequency = fftfreq(size, 1/sample_rate)
    time = (np.arange(size)-pad) / sample_rate
    edges = np.linspace(low_hz, high_hz, bands+1)
    offsets = np.zeros(1) if bands == 1 else np.random.default_rng(seed).uniform(0, 2*np.pi, bands)
    delta = np.zeros_like(padded)
    for low, high, offset in zip(edges[:-1], edges[1:], offsets):
        mask = (frequency >= low) & (frequency < high)
        carrier = ifft(spectrum * (2*mask)[:, None], axis=0)
        phase = depth_radians * np.sin(2*np.pi*rate_hz*time + offset)
        delta += (carrier * np.expm1(1j*phase)[:, None]).real
    result = channels + delta[pad:pad+len(x)]
    gain = 1.0
    peak = float(np.abs(result).max())
    if peak_limit is not None and peak > peak_limit:
        gain = peak_limit / peak
        result *= gain
    return (result[:, 0] if mono else result), gain
