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


def reconstruct(model, features, audio, sample_rate, frequency_sigma, time_sigma):
    import torch
    if (audio.ndim not in (1, 2) or not len(audio) or
            (audio.ndim == 2 and audio.shape[1] not in (1, 2)) or
            not np.isfinite(audio).all() or not 8000 <= sample_rate <= 96000):
        raise ValueError('Use finite, nonempty mono/stereo audio at 8–96 kHz')
    if not all(np.isfinite(s) and 0 <= s <= 3 for s in (frequency_sigma, time_sigma)):
        raise ValueError('Each smoothing sigma must be finite and between zero and three')
    channels = audio[:, None] if audio.ndim == 1 else audio
    parameters, outputs = model.h, []
    divisor = gcd(sample_rate, parameters.sampling_rate)
    with torch.inference_mode():
        for channel in channels.T:
            native = resample_poly(channel, parameters.sampling_rate // divisor,
                                   sample_rate // divisor)
            padded = np.pad(native, (0, (-len(native)) % parameters.hop_size))
            mel = features(torch.as_tensor(padded)[None], parameters)
            smoothed = gaussian_filter(mel.numpy(), sigma=(0, frequency_sigma, time_sigma),
                                       mode='reflect')
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
    args = parser.parse_args()
    if not all(np.isfinite(s) and 0 <= s <= 3 for pair in args.sigmas for s in pair):
        parser.error('Each smoothing sigma must be finite and between zero and three')
    if len({tuple(pair) for pair in args.sigmas}) != len(args.sigmas):
        parser.error('Duplicate settings are not separate experiments')
    source_sha = digest(args.source)
    audio, rate = sf.read(args.source, dtype='float32')
    args.directory.mkdir(parents=True, exist_ok=False)
    report = {'source_sha256': source_sha, 'probe_sha256': digest(Path(__file__)),
              'upstream_assets': ASSETS, 'reference_text_used': False,
              'detector_requests': 0, 'quality_verified': False, 'complete': False,
              'settings': {'gaussian_mode': 'reflect', 'flac_subtype': 'PCM_24',
                           'torch_threads': 4, 'mel_chunk_frames': 256, 'context_frames': 32},
              'candidates': []}

    def save():
        path = args.directory / 'generation.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    save()
    model, features = load_vocoder()
    for frequency_sigma, time_sigma in args.sigmas:
        started = time.perf_counter()
        rebuilt, gain = reconstruct(model, features, audio, rate, frequency_sigma, time_sigma)
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
        (folder / 'generation.json').write_text(json.dumps(row, indent=2) + '\n')
        report['candidates'].append(row)
        save()
        print('VOCODER_RENDERED', json.dumps(row), flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
