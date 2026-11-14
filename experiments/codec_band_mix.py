"""Mix complete codec/vocoder reconstructions for the band-replacement probe.

This processes two already reconstructed recordings, with matching sample grids.
It does not load a model, evaluate speech quality, or check a watermark detector.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
import soundfile as sf


def mix_reconstructions(vocoder, codec, sample_rate, low=1500, high=3500):
    """Replace one band without moving samples or combining stereo channels."""
    vocoder, codec = np.asarray(vocoder), np.asarray(codec)
    if (vocoder.shape != codec.shape or vocoder.ndim not in (1, 2) or
            (vocoder.ndim == 2 and vocoder.shape[1] not in (1, 2))):
        raise ValueError('Reconstructions must have the same complete mono/stereo shape')
    if (not isinstance(sample_rate, (int, np.integer)) or not 8000 <= sample_rate <= 96000 or
            not 5*sample_rate <= len(vocoder) <= 300*sample_rate):
        raise ValueError('Expected complete 5–300-second recordings at 8–96 kHz')
    if not np.isfinite(vocoder).all() or not np.isfinite(codec).all():
        raise ValueError('Reconstructions contain nonfinite samples')
    if not 0 < low < high < sample_rate/2:
        raise ValueError('Band must lie strictly below the Nyquist frequency')
    vocoder = np.asarray(vocoder, dtype=np.float32)
    codec = np.asarray(codec, dtype=np.float32)
    sos = butter(6, [low, high], btype='bandpass', fs=sample_rate, output='sos')
    output = np.asarray(vocoder+sosfiltfilt(sos, codec-vocoder, axis=0), dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError('Band replacement produced nonfinite samples')
    gain = min(1., .999/max(1e-12, float(np.max(np.abs(output)))))
    return output*gain, gain


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vocoder', type=Path, help='Complete vocoder reconstruction')
    parser.add_argument('codec', type=Path, help='Complete codec reconstruction, same rate/count/channels')
    parser.add_argument('output', type=Path, help='New PCM24 FLAC file')
    parser.add_argument('--low', type=float, default=1500)
    parser.add_argument('--high', type=float, default=3500)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    if args.metadata and args.output.resolve() == args.metadata.resolve():
        parser.error('Output and metadata must use different paths')
    if args.output.exists() or (args.metadata and args.metadata.exists()):
        parser.error('Output and metadata paths must be new files')
    checksums = {role: digest(path) for role, path in [('vocoder', args.vocoder), ('codec', args.codec)]}
    vocoder, rate = sf.read(args.vocoder, dtype='float32')
    codec, codec_rate = sf.read(args.codec, dtype='float32')
    if rate != codec_rate:
        parser.error('Reconstructions have different sample rates; no implicit resampling is performed')
    output, gain = mix_reconstructions(vocoder, codec, rate, args.low, args.high)
    if checksums != {role: digest(path) for role, path in [('vocoder', args.vocoder), ('codec', args.codec)]}:
        raise ValueError('An input file changed during processing')
    with args.output.open('xb') as handle:
        sf.write(handle, output, rate, format='FLAC', subtype='PCM_24')
    report = {'method': 'experimental_codec_vocoder_band_mix', 'input_sha256': checksums,
              'output_sha256': digest(args.output), 'sample_rate': rate, 'samples': len(output),
              'channels': 1 if output.ndim == 1 else output.shape[1], 'gain': gain,
              'band_hz': [args.low, args.high], 'filter_order': 6, 'zero_phase': True,
              'watermark_status': 'not_checked', 'quality_verified': False,
              'recognized_transcripts_included': False}
    if args.metadata:
        with args.metadata.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
