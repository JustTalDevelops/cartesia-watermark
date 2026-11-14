"""Whole-file phase-only band-delay probe; no models or detector requests.

The frequency response has unit magnitude before padding/cropping, peak gain
and PCM24 encoding. This is a development transformation, not an erasure claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

DEFAULT_EDGES = (1250., 1500., 3500., 3750.)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def phase_response(frequencies, delay_ms, edges=DEFAULT_EDGES):
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if (frequencies.ndim != 1 or len(frequencies) < 2 or
            not np.isfinite(frequencies).all() or frequencies[0] < 0 or
            not (np.diff(frequencies) > 0).all()):
        raise ValueError('Use a finite increasing frequency grid')
    if (len(edges) != 4 or not np.isfinite(edges).all() or
            not 0 <= edges[0] < edges[1] < edges[2] < edges[3] <= frequencies[-1]):
        raise ValueError('Use four increasing band edges below Nyquist')
    if not np.isfinite(delay_ms) or not 0 <= delay_ms <= 100:
        raise ValueError('Use a finite delay from zero to 100 milliseconds')
    low, rise, fall, high = edges
    weights = np.minimum(np.clip((frequencies-low)/(rise-low), 0, 1),
                         np.clip((high-frequencies)/(high-fall), 0, 1))
    # Integrate the desired group delay. Multiplying f by a band mask instead
    # would introduce large, unwanted delays at the mask transitions.
    area = np.r_[0., np.cumsum((weights[:-1]+weights[1:])*.5*np.diff(frequencies))]
    return np.exp(-2j*np.pi*(delay_ms/1000)*area)


def transform(audio, rate, delay_ms, edges=DEFAULT_EDGES):
    if (audio.ndim not in (1, 2) or len(audio) < 2 or not np.isfinite(audio).all() or
            (audio.ndim == 2 and audio.shape[1] not in (1, 2)) or not 8000 <= rate <= 96000):
        raise ValueError('Use finite mono/stereo audio at 8–96 kHz')
    mono = audio.ndim == 1
    channels = audio[:, None] if mono else audio
    padding = min(rate, len(audio)-1)
    padded = np.pad(channels, ((padding, padding), (0, 0)), mode='reflect')
    fft_size = 1 << (len(padded)-1).bit_length()
    frequencies = np.fft.rfftfreq(fft_size, 1/rate)
    response = phase_response(frequencies, delay_ms, edges)
    response[-1] = 1.  # Keep the real Nyquist bin real and at unit magnitude.
    spectrum = np.fft.rfft(padded, n=fft_size, axis=0)
    rebuilt = np.fft.irfft(spectrum*response[:, None], n=fft_size, axis=0)[padding:padding+len(audio)]
    if rebuilt.shape != channels.shape or not np.isfinite(rebuilt).all():
        raise RuntimeError('Phase transform changed the whole recording grid')
    peak = float(np.abs(rebuilt).max())
    # An already-quantized .999 peak can round upward by one PCM24 step.
    # Do not change the whole recording merely to undo that harmless rounding.
    gain = 1. if peak <= .999+2**-23 else .999/max(peak, 1e-20)
    result = (rebuilt*gain).astype(np.float32)
    return (result[:, 0] if mono else result), gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--delays-ms', type=float, nargs='+', default=[0., 33., 50.])
    args = parser.parse_args()
    if (len(set(args.delays_ms)) != len(args.delays_ms) or
            not all(np.isfinite(v) and 0 <= v <= 100 for v in args.delays_ms)):
        parser.error('Use unique finite delays from zero to 100 milliseconds')
    source_sha = digest(args.source)
    audio, rate = sf.read(args.source, dtype='float32')
    if not 5 <= len(audio)/rate <= 300:
        parser.error('Use a complete five-to-three-hundred-second development recording')
    args.directory.mkdir(parents=True, exist_ok=False)
    report = {'processing_input_sha256': source_sha, 'probe_sha256': digest(Path(__file__)),
              'reference_text_used': False, 'detector_requests': 0, 'quality_verified': False,
              'band_hz': DEFAULT_EDGES, 'padding': 'One-second reflection on both sides, then power-of-two FFT',
              'nyquist_policy': 'Real unit response at the Nyquist bin',
              'peak_gain_policy': 'No gain below .999 plus one PCM24 step; otherwise limit to .999',
              'pcm24_rounding': 'Nearest-even rounding before the soundfile writer',
              'numpy': np.__version__, 'soundfile': sf.__version__,
              'candidates': [], 'complete': False}
    out = args.directory/'generation.json'

    def save():
        temporary = out.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(out)

    save()
    for delay in args.delays_ms:
        started = time.perf_counter()
        rebuilt, gain = transform(audio, rate, delay)
        if digest(args.source) != source_sha:
            raise ValueError('Processing input changed')
        folder = args.directory/f'{delay:g}ms'
        folder.mkdir()
        candidate = folder/'candidate.flac'
        canonical = np.rint(rebuilt*np.float32(2**23))/np.float32(2**23)
        with candidate.open('xb') as handle:
            sf.write(handle, canonical, rate, format='FLAC', subtype='PCM_24')
        row = {'processing_input_sha256': source_sha, 'output_sha256': digest(candidate),
               'delay_ms': delay, 'band_hz': DEFAULT_EDGES, 'gain': gain, 'samples': len(audio),
               'sample_rate': rate, 'channels': 1 if audio.ndim == 1 else audio.shape[1],
               'elapsed_seconds': time.perf_counter()-started, 'complete': True, 'quality_verified': False}
        (folder/'generation.json').write_text(json.dumps(row, indent=2)+'\n')
        report['candidates'].append(row)
        save()
        print('PHASE_TRANSFORMED', json.dumps(row), flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
