"""Experimental speech reconstruction through a fixed, smoothed BigVGAN prior.

Local inference only. Detector clearance and speech fidelity require separate
evaluation. Upstream source and weights must be installed before processing.
"""
import argparse
from contextlib import redirect_stdout
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

from vocoder_inference import synthesize_chunks

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT/'.cache/bigvgan'
MODEL_DIR = ROOT/'.cache/bigvgan-model'
ASSETS = json.loads((Path(__file__).parent/'vocoder_assets.json').read_text())
PRESET = 'smoothed-vocoder'
SETTINGS = {'mel_sigma_frequency': 1., 'mel_sigma_time': 1., 'gaussian_mode': 'reflect',
            'mel_chunk_frames': 256, 'context_frames': 32, 'torch_threads': 4}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_vocoder():
    for directory, expected in ((SOURCE, ASSETS['source_sha256']),
                                (MODEL_DIR, ASSETS['model_sha256'])):
        for name, sha in expected.items():
            if not (directory/name).is_file() or digest(directory/name) != sha:
                raise RuntimeError(f'Missing or changed pinned vocoder asset: {name}')
    sys.path.insert(0, str(SOURCE))
    import torch
    from bigvgan import BigVGAN, load_hparams_from_json
    from meldataset import get_mel_spectrogram
    torch.set_num_threads(4)
    parameters = load_hparams_from_json(MODEL_DIR/'config.json')
    model = BigVGAN(parameters, use_cuda_kernel=False)
    state = torch.load(MODEL_DIR/'bigvgan_generator.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['generator'])
    model.remove_weight_norm()
    model.eval().requires_grad_(False)
    return model, get_mel_spectrogram


def reconstruct(model, features, audio, sample_rate):
    import torch
    if (audio.ndim not in (1, 2) or not len(audio) or
            (audio.ndim == 2 and audio.shape[1] not in (1, 2)) or
            not np.isfinite(audio).all() or not 8000 <= sample_rate <= 96000):
        raise ValueError('Choose finite, nonempty mono/stereo audio at 8–96 kHz')
    mono = audio.ndim == 1
    channels = audio[:, None] if mono else audio
    parameters, outputs = model.h, []
    divisor = gcd(sample_rate, parameters.sampling_rate)
    with torch.inference_mode():
        for channel in channels.T:
            native = resample_poly(channel, parameters.sampling_rate//divisor, sample_rate//divisor)
            padded = np.pad(native, (0, (-len(native)) % parameters.hop_size))
            mel = features(torch.as_tensor(padded)[None], parameters)
            mel = torch.from_numpy(gaussian_filter(mel.numpy(), sigma=(0, 1., 1.), mode='reflect'))
            rebuilt = synthesize_chunks(mel, parameters.hop_size,
                                        lambda chunk: model(chunk).flatten().numpy())[:len(native)]
            y = resample_poly(rebuilt, sample_rate//divisor, parameters.sampling_rate//divisor)[:len(audio)]
            if len(y) != len(audio) or not np.isfinite(y).all():
                raise RuntimeError('Vocoder did not preserve recording length')
            outputs.append(y)
    result = np.stack(outputs, axis=1)
    gain = min(1., .999/max(float(np.abs(result).max()), 1e-20))
    result = result*gain
    return (result[:, 0] if mono else result), gain


def validate_paths(source, output, metadata):
    if source.resolve() == output.resolve() or output.exists() or output.suffix.lower() != '.wav':
        raise ValueError('Choose a new WAV output separate from the input')
    if metadata and (metadata.exists() or metadata.resolve() in (source.resolve(), output.resolve())):
        raise ValueError('Choose a new metadata file separate from the audio files')


def process_file(model, features, source, output, preset=PRESET, metadata=None):
    validate_paths(source, output, metadata)
    if preset != PRESET:
        raise ValueError('Unknown vocoder preset')
    start = time.monotonic()
    audio, rate = sf.read(source, dtype='float32')
    result, gain = reconstruct(model, features, audio, rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        sf.write(handle, result, rate, format='WAV', subtype='FLOAT')
    report = {'preset': PRESET, 'settings': SETTINGS, 'source_commit': ASSETS['source_commit'],
              'model': ASSETS['model_id'], 'model_revision': ASSETS['model_revision'],
              'checkpoint_sha256': ASSETS['model_sha256']['bigvgan_generator.pt'],
              'sample_rate': rate, 'samples': len(result),
              'channels': 1 if result.ndim == 1 else result.shape[1], 'gain': gain,
              'elapsed_seconds': time.monotonic()-start, 'output_sha256': digest(output)}
    if metadata:
        with metadata.open('x') as handle:
            json.dump(report, handle, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.input or args.output or args.metadata or args.check:
            parser.error('Worker mode does not accept file arguments')
    elif not args.check:
        if not args.input or not args.output:
            parser.error('Input and output paths are required')
        try:
            validate_paths(args.input, args.output, args.metadata)
        except ValueError as error:
            parser.error(str(error))
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        model, features = load_vocoder()
        if args.worker:
            print(json.dumps({'ready': True}), file=protocol, flush=True)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    report = process_file(model, features, Path(request['input']), Path(request['output']),
                                          request['preset'], Path(request['metadata']) if request.get('metadata') else None)
                    reply = {'report': report}
                except Exception as error:
                    reply = {'error': f'{type(error).__name__}: {error}'}
                print(json.dumps(reply), file=protocol, flush=True)
            return
        if args.check:
            report = {'ready': True, 'checkpoint_sha256': ASSETS['model_sha256']['bigvgan_generator.pt']}
        else:
            report = process_file(model, features, args.input, args.output, metadata=args.metadata)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
