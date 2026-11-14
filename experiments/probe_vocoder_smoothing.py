"""Render a fixed anisotropic mel-smoothing experiment, without detector calls.

This uses the pinned BigVGAN assets and chunk renderer. It does not change the
production preset, perform speech recognition, or claim watermark removal.
"""
import argparse
import hashlib
import json
from math import gcd
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from reconstruct_vocoder import ASSETS, digest, load_vocoder
from vocoder_inference import synthesize_chunks


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--sigmas', type=float, nargs=2, action='append', required=True,
                        metavar=('FREQUENCY', 'TIME'))
    parser.add_argument('--band-hz', type=float, nargs=4,
                        metavar=('START', 'PLATEAU_START', 'PLATEAU_END', 'END'),
                        help='Blend strong smoothing inside this fixed band with sigma .5/.5 elsewhere')
    args = parser.parse_args()
    if not all(np.isfinite(s) and 0 <= s <= 3 for pair in args.sigmas for s in pair):
        parser.error('Each smoothing sigma must be finite and between zero and three')
    if len({tuple(pair) for pair in args.sigmas}) != len(args.sigmas):
        parser.error('Duplicate settings are not separate experiments')
    if args.band_hz is not None:
        try:
            band_weights(np.array([0.]), args.band_hz)
        except ValueError as error:
            parser.error(str(error))
    source_sha = digest(args.source)
    audio, rate = sf.read(args.source, dtype='float32')
    args.directory.mkdir(parents=True, exist_ok=False)
    report = {'source_sha256': source_sha, 'probe_sha256': digest(Path(__file__)),
              'upstream_assets': ASSETS, 'reference_text_used': False,
              'detector_requests': 0, 'quality_verified': False, 'complete': False,
              'settings': {'gaussian_mode': 'reflect', 'flac_subtype': 'PCM_24',
                           'torch_threads': 4, 'mel_chunk_frames': 256, 'context_frames': 32},
              'candidates': []}
    if args.band_hz is not None:
        report['settings'].update(band_hz=args.band_hz, outside_band_sigmas=[.5, .5],
                                  band_domain='Log-mel features; actual Slaney center frequencies')

    def save():
        path = args.directory / 'generation.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    save()
    model, features = load_vocoder()
    for frequency_sigma, time_sigma in args.sigmas:
        started = time.perf_counter()
        rebuilt, gain = reconstruct(model, features, audio, rate, frequency_sigma, time_sigma, args.band_hz)
        if digest(args.source) != source_sha:
            raise ValueError('Source changed during rendering')
        folder = args.directory / f'f{frequency_sigma:g}_t{time_sigma:g}'
        folder.mkdir()
        path = folder / 'candidate.flac'
        with path.open('xb') as handle:
            sf.write(handle, rebuilt, rate, format='FLAC', subtype='PCM_24')
        row = {'source_sha256': source_sha, 'output_sha256': digest(path),
               'sigma_frequency': frequency_sigma, 'sigma_time': time_sigma,
               'gain': gain, 'samples': len(audio), 'sample_rate': rate,
               'channels': 1 if audio.ndim == 1 else audio.shape[1],
               'render_seconds': time.perf_counter() - started, 'complete': True,
               'quality_verified': False}
        if args.band_hz is not None:
            row.update(band_hz=args.band_hz, outside_band_sigmas=[.5, .5])
        (folder / 'generation.json').write_text(json.dumps(row, indent=2) + '\n')
        report['candidates'].append(row)
        save()
        print('VOCODER_RENDERED', json.dumps(row), flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
