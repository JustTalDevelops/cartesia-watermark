"""Internal unscaled donor renderer for speech repair, never a final recording.

Input is a JSON manifest with source, source_sha256 and regions_seconds. The
intermediate WAV contains zeros in omitted blocks; only the requested speech
regions and their resampling context are valid. No detector requests.
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
from scipy.ndimage import gaussian_filter
from scipy.signal import resample_poly
import soundfile as sf

import reconstruct_vocoder as reference
from vocoder_regions import synthesize_regions

ROOT = Path(__file__).resolve().parent.parent
PRESET = 'regional-vocoder'
PINNED_FILES = {
    'reconstruct_vocoder.py': '3378845287c3f1bf874147392184c5b15b43ca954adacd6eac484715e4564db0',
    'vocoder_inference.py': '38d02df4aa5d488aab7b37320dcbf6cc6e980178e23721783145adef138c37f2',
    'vocoder_assets.json': 'ea7870e32da6336a800a3ddbe555d527267124470b44f8e7e0c72603aae4e7f6',
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_vocoder():
    for name, expected in PINNED_FILES.items():
        if digest(ROOT/'tools'/name) != expected:
            raise RuntimeError('The pinned full vocoder implementation changed')
    return reference.load_vocoder()


def reconstruct_regions(model, features, audio, sample_rate, intervals):
    import torch
    if (audio.ndim not in (1, 2) or not len(audio) or not np.isfinite(audio).all() or
            (audio.ndim == 2 and audio.shape[1] not in (1, 2)) or not 8000 <= sample_rate <= 96000):
        raise ValueError('Choose finite, nonempty mono/stereo audio at 8–96 kHz')
    duration = len(audio)/sample_rate
    if (not intervals or any(len(r) != 2 or not all(np.isfinite(v) for v in r) or
                             not 0 <= r[0] < r[1] <= duration+1e-9 for r in intervals)):
        raise ValueError('Choose nonempty speech intervals inside the recording')
    mono = audio.ndim == 1
    channels = audio[:, None] if mono else audio
    parameters = model.h
    divisor = gcd(sample_rate, parameters.sampling_rate)
    outputs, channel_blocks, total_blocks = [], [], None
    with torch.inference_mode():
        for channel in channels.T:
            native = resample_poly(channel, parameters.sampling_rate//divisor, sample_rate//divisor)
            padded = np.pad(native, (0, (-len(native)) % parameters.hop_size))
            mel = features(torch.as_tensor(padded)[None], parameters)
            mel = torch.from_numpy(gaussian_filter(mel.numpy(), sigma=(0, 1., 1.), mode='reflect'))
            # Ten milliseconds comfortably covers resample_poly's default FIR
            # support over the supported 8–96 kHz rate range.
            guard = int(np.ceil(.01*parameters.sampling_rate))
            regions = [(max(0, int(np.floor(start*parameters.sampling_rate))-guard),
                        min(len(native), int(np.ceil(end*parameters.sampling_rate))+guard))
                       for start, end in intervals]
            rebuilt, blocks = synthesize_regions(
                mel, parameters.hop_size, lambda chunk: model(chunk).flatten().numpy(), regions)
            rebuilt = rebuilt[:len(native)]
            result = resample_poly(rebuilt, sample_rate//divisor, parameters.sampling_rate//divisor)[:len(audio)]
            if len(result) != len(audio) or not np.isfinite(result).all():
                raise RuntimeError('Regional vocoder did not preserve recording length')
            outputs.append(result)
            channel_blocks.append(list(blocks))
            total_blocks = (mel.shape[-1]+255)//256
    result = np.stack(outputs, axis=1)
    return (result[:, 0] if mono else result), {
        'rendered_blocks_per_channel': channel_blocks, 'full_blocks_per_channel': total_blocks,
        'resampling_guard_seconds': .01, 'donor_gain': 1., 'donor_is_unscaled': True,
        'waveform_is_partial': True, 'valid_regions_seconds': intervals}


def process_file(model, features, manifest_path, output, preset=PRESET, metadata=None):
    reference.validate_paths(manifest_path, output, metadata)
    if preset != PRESET:
        raise ValueError('Unknown regional vocoder preset')
    manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['source'])
    if digest(source) != manifest['source_sha256']:
        raise RuntimeError('Donor source changed')
    started = time.monotonic()
    audio, rate = sf.read(source, dtype='float32')
    rebuilt, details = reconstruct_regions(model, features, audio, rate, manifest['regions_seconds'])
    if digest(source) != manifest['source_sha256'] or digest(manifest_path) != manifest_sha:
        raise RuntimeError('Donor source or manifest changed during processing')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        sf.write(handle, rebuilt, rate, format='WAV', subtype='FLOAT')
    report = {'preset': PRESET, 'settings': reference.SETTINGS, **details,
              'source_sha256': manifest['source_sha256'], 'input_manifest_sha256': manifest_sha,
              'source_commit': reference.ASSETS['source_commit'],
              'checkpoint_sha256': reference.ASSETS['model_sha256']['bigvgan_generator.pt'],
              'sample_rate': rate, 'samples': len(rebuilt), 'channels': 1 if rebuilt.ndim == 1 else rebuilt.shape[1],
              'elapsed_seconds': time.monotonic()-started, 'output_sha256': digest(output)}
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
            parser.error('Input manifest and output paths are required')
        try:
            reference.validate_paths(args.input, args.output, args.metadata)
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
        report = {'ready': True, 'preset': PRESET} if args.check else process_file(
            model, features, args.input, args.output, metadata=args.metadata)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
