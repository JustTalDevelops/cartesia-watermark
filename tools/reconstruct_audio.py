"""Local speech reconstruction using the listener-approved reference preset.

Run with the isolated reconstruction environment, not the web environment:
    .cache/resemble-venv/bin/python tools/reconstruct_audio.py input.wav output.wav

No detector requests, training, or remote audio processing. The preset is not a
guarantee of watermark removal. All required model assets must already exist.
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
from scipy.signal import resample_poly
import torch

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / '.cache/resemble-enhance'
MODEL_DIR = ROOT / '.cache/resemble-model/enhancer_stage2'
CHECKPOINT_SHA256 = 'f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6'
PRESETS = {
    'standard': dict(nfe=32, solver='midpoint', lambd=.1, tau=.5),
    'full-flow': dict(nfe=32, solver='midpoint', lambd=.1, tau=1.),
    'strong': dict(nfe=32, solver='midpoint', lambd=.9, tau=1.),
}


def load_reconstructor(preset, boundary_safe=False):
    if preset not in PRESETS:
        raise ValueError('Unknown reconstruction preset')
    for relative in ['hparams.yaml', 'ds/G/latest', 'ds/G/default/mp_rank_00_model_states.pt']:
        if not (MODEL_DIR / relative).is_file():
            raise RuntimeError('Required reconstruction model assets are missing')
    checkpoint = MODEL_DIR / 'ds/G/default/mp_rank_00_model_states.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError('Reconstruction checkpoint differs from the pinned model')
    sys.path.insert(0, str(SOURCE))
    from resemble_enhance.enhancer.inference import load_enhancer
    from resemble_enhance.inference import inference
    torch.set_num_threads(4)
    model = load_enhancer(MODEL_DIR, 'cpu')
    model.configurate_(**PRESETS[preset])
    if boundary_safe:
        from reconstruction_inference import inference
    return model, inference


def reconstruct(model, inference, audio, sample_rate, seed=2035):
    mono = audio.ndim == 1
    source = audio[:, None] if mono else audio
    if source.ndim != 2 or source.shape[1] not in (1, 2) or not len(source):
        raise ValueError('Choose a nonempty mono or stereo recording')
    if not np.isfinite(source).all() or not 8000 <= sample_rate <= 96000:
        raise ValueError('Invalid audio samples or sample rate')
    outputs = []
    for channel in range(source.shape[1]):
        torch.manual_seed(seed)
        wave = torch.from_numpy(np.ascontiguousarray(source[:, channel], dtype=np.float32))
        rebuilt, rate = inference(model, wave, sample_rate, 'cpu', chunk_seconds=10., overlap_seconds=1.)
        y = rebuilt.cpu().numpy()
        if rate != sample_rate:
            divisor = gcd(rate, sample_rate)
            y = resample_poly(y, sample_rate // divisor, rate // divisor)
        tolerance = max(8, int(np.ceil(sample_rate / 44100)) * 4)
        if abs(len(y) - len(source)) > tolerance or not np.isfinite(y).all():
            raise RuntimeError('Reconstruction did not preserve the recording duration')
        y = np.pad(y[:len(source)], (0, max(0, len(source) - len(y))))
        outputs.append(y)
    result = np.stack(outputs, axis=1)
    # A shared gain preserves the stereo balance and prevents PCM previews
    # from clipping. This is the same peak rule as the approved mono trial.
    gain = min(1., .999 / max(float(np.abs(result).max()), 1e-20))
    result = result * gain
    return (result[:, 0] if mono else result), gain


def validate_paths(source, output, metadata):
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Choose a new output file, separate from the input')
    if output.suffix.lower() != '.wav':
        raise ValueError('Use a WAV output file')
    if metadata and (metadata.exists() or metadata.resolve() in (source.resolve(), output.resolve())):
        raise ValueError('Choose a new metadata file, separate from the audio files')


def process_file(model, inference, source, output, preset, metadata=None, started=None):
    validate_paths(source, output, metadata)
    if preset not in PRESETS:
        raise ValueError('Unknown reconstruction preset')
    started = time.monotonic() if started is None else started
    model.configurate_(**PRESETS[preset])
    audio, rate = sf.read(source, dtype='float32')
    result, gain = reconstruct(model, inference, audio, rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        sf.write(handle, result, rate, format='WAV', subtype='FLOAT')
    report = dict(preset=preset, settings=PRESETS[preset], seed=2035,
                  inference_implementation=f'{inference.__module__}.{inference.__qualname__}',
                  checkpoint_sha256=CHECKPOINT_SHA256, sample_rate=rate,
                  samples=len(result), channels=1 if result.ndim == 1 else result.shape[1],
                  gain=gain, elapsed_seconds=time.monotonic()-started,
                  output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    if metadata:
        with metadata.open('x') as handle:
            handle.write(json.dumps(report, indent=2))
    return report


def worker_main(boundary_safe=False):
    """Private line-delimited JSON pipe; keep the model out of the web runtime."""
    protocol = sys.stdout
    # Upstream imports and inference log progress. Reserve stdout for replies.
    with redirect_stdout(sys.stderr):
        model, inference = load_reconstructor('standard', boundary_safe=boundary_safe)
        print(json.dumps({'ready': True}), file=protocol, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                metadata = Path(request['metadata']) if request.get('metadata') else None
                report = process_file(model, inference, Path(request['input']), Path(request['output']),
                                      request['preset'], metadata)
                reply = {'report': report}
            except Exception as error:
                reply = {'error': f'{type(error).__name__}: {error}'}
            print(json.dumps(reply), file=protocol, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--preset', choices=tuple(PRESETS), default='standard')
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--boundary-safe', action='store_true',
                        help='Experimental chunk scheduling that avoids redundant tail chunks')
    args = parser.parse_args()
    if args.worker:
        if args.input or args.output or args.metadata or args.check:
            parser.error('Worker mode does not accept file arguments')
        worker_main(boundary_safe=args.boundary_safe)
        return
    if not args.check:
        if not args.input or not args.output:
            parser.error('Input and output paths are required')
        try:
            validate_paths(args.input, args.output, args.metadata)
        except ValueError as error:
            parser.error(str(error))
    started = time.monotonic()
    model, inference = load_reconstructor(args.preset, boundary_safe=args.boundary_safe)
    if args.check:
        print(json.dumps({'ready': True, 'checkpoint_sha256': CHECKPOINT_SHA256}))
        return
    report = process_file(model, inference, args.input, args.output, args.preset, args.metadata, started)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
