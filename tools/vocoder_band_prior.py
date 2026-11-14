"""Fixed band-focused BigVGAN prior; experimental, without a removal guarantee.

The math is copied unchanged from the c6cae6b smoothing prototype. The worker
fixes sigma 1/1 inside the 1–4 kHz trapezoid, with sigma .5/.5 outside it.
"""
from math import gcd
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import resample_poly
from vocoder_inference import synthesize_chunks

BAND_EDGES_HZ = (1000., 1500., 3500., 4000.)

def band_weights(centers, edges):
    """A fixed trapezoid on actual Slaney mel-bin centers, not FFT indices."""
    centers = np.asarray(centers)
    if (len(edges) != 4 or not np.isfinite(edges).all() or
            not 0 <= edges[0] < edges[1] < edges[2] < edges[3]):
        raise ValueError('Use four finite increasing nonnegative band edges')
    low, rise, fall, high = edges
    return np.minimum(np.clip((centers-low)/(rise-low), 0, 1),
                      np.clip((high-centers)/(high-fall), 0, 1)).astype(np.float32)

def smooth_mel(values, frequency_sigma, time_sigma, centers=None, band_edges=None):
    strong = gaussian_filter(values, sigma=(0, frequency_sigma, time_sigma), mode='reflect')
    if band_edges is None:
        return strong
    if centers is None or len(centers) != values.shape[1]:
        raise ValueError('Provide the actual center frequency of every mel bin')
    gentle = gaussian_filter(values, sigma=(0, .5, .5), mode='reflect')
    weights = band_weights(centers, band_edges)[None, :, None]
    return np.where(weights == 0, gentle,
                    np.where(weights == 1, strong, gentle+(strong-gentle)*weights))

def reconstruct(model, features, audio, sample_rate, frequency_sigma, time_sigma, band_edges=None):
    import torch
    if (audio.ndim not in (1, 2) or not len(audio) or
            (audio.ndim == 2 and audio.shape[1] not in (1, 2)) or
            not np.isfinite(audio).all() or not 8000 <= sample_rate <= 96000):
        raise ValueError('Use finite, nonempty mono/stereo audio at 8–96 kHz')
    if not all(np.isfinite(s) and 0 <= s <= 3 for s in (frequency_sigma, time_sigma)):
        raise ValueError('Each smoothing sigma must be finite and between zero and three')
    channels = audio[:, None] if audio.ndim == 1 else audio
    parameters, outputs = model.h, []
    centers = None
    if band_edges is not None:
        from librosa import mel_frequencies
        if band_edges[-1] > parameters.sampling_rate/2:
            raise ValueError('Band edges exceed the vocoder Nyquist frequency')
        centers = mel_frequencies(n_mels=parameters.num_mels+2, fmin=parameters.fmin,
                                  fmax=parameters.fmax or parameters.sampling_rate/2)[1:-1]
        band_weights(centers, band_edges)
    divisor = gcd(sample_rate, parameters.sampling_rate)
    with torch.inference_mode():
        for channel in channels.T:
            native = resample_poly(channel, parameters.sampling_rate // divisor,
                                   sample_rate // divisor)
            padded = np.pad(native, (0, (-len(native)) % parameters.hop_size))
            mel = features(torch.as_tensor(padded)[None], parameters)
            smoothed = smooth_mel(mel.numpy(), frequency_sigma, time_sigma, centers, band_edges)
            rebuilt = synthesize_chunks(torch.from_numpy(smoothed), parameters.hop_size,
                                        lambda chunk: model(chunk).flatten().numpy())[:len(native)]
            y = resample_poly(rebuilt, sample_rate // divisor,
                              parameters.sampling_rate // divisor)[:len(audio)]
            if len(y) != len(audio) or not np.isfinite(y).all():
                raise RuntimeError('Vocoder did not preserve recording length')
            outputs.append(y)
    result = np.stack(outputs, axis=1)
    gain = min(1., .999 / max(float(np.abs(result).max()), 1e-20))
    result *= gain
    return (result[:, 0] if audio.ndim == 1 else result), gain
