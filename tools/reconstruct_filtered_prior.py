"""Experimental reconstruction with a band-limited starting prior.

The original full-band speech condition is retained. Only the waveform used
to encode the starting prior is low-passed. Local inference; no API requests.
Requires the separately installed pinned Resemble runtime and assets.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt
import soundfile as sf
import torch

import reconstruct_audio as reference

ROOT = Path(__file__).resolve().parent.parent
PRESET = 'filtered-prior'
SOURCE_COMMIT = '8e978149bfe8abab3eb77d965d579a111afdb0ff'
SETTINGS = {'nfe': 32, 'solver': 'midpoint', 'denoising_blend': .1,
            'prior_temperature': .8, 'prior_lowpass_hz': 2300., 'filter_order': 8,
            'filter': 'Butterworth SOS forward/backward, float64; normalized native chunk prior only',
            'conditioning': 'original full-band speech', 'boundary_safe': True,
            'threads': 4}
PINNED_FILES = {
    ROOT/'tools/reconstruct_audio.py': '25f3e120598995e398fe76a68a3fda943b53e4f294cad5887207888fb9229c96',
    ROOT/'tools/reconstruction_inference.py': '1a3c6b8ae05019f04b9650a11b81a331b4ef6b11e9dc43c34e95a0371cbf8a8f',
    reference.MODEL_DIR/'hparams.yaml': '80c3f15bc5a5b2cacf2c698699a0f6599d62911c0d53e1d6dee895c0d7cbaeac',
    reference.SOURCE/'resemble_enhance/enhancer/enhancer.py': '6e6d6ece0df0f02d94c3fdf12ddd6d569f36bfd21d89f0bc81b7257166a66670',
    reference.SOURCE/'resemble_enhance/enhancer/lcfm/lcfm.py': '97ed3982a8c94a7070c9a3295f764610387191776a13fd504279abdcdd475dd5',
    reference.SOURCE/'resemble_enhance/melspec.py': 'b80ddfad3e0cb10a583eead8436b63729803e3990b6d96601104d688105c32e5',
    reference.SOURCE/'resemble_enhance/common.py': '675da1829bea4d8cac20a253e452dfbe546599c6886f99979d43be2578fa3916',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FilteredPrior(torch.nn.Module):
    """Supply an alternate prior without modifying upstream conditioning code."""
    def __init__(self, enhancer):
        super().__init__()
        self.enhancer, self.hp = enhancer, enhancer.hp
        self.sos = butter(SETTINGS['filter_order'], SETTINGS['prior_lowpass_hz'],
                          fs=self.hp.wav_rate, output='sos')
        self._prior = None
        self._lock = threading.RLock()
        self._hook = enhancer.lcfm.register_forward_pre_hook(self._replace_prior, with_kwargs=True)

    def _replace_prior(self, module, args, kwargs):
        if self._prior is None or 'ψ0' not in kwargs or kwargs['ψ0'].shape != self._prior.shape:
            raise RuntimeError('The pinned model did not supply the expected starting prior')
        return args, {**kwargs, 'ψ0': self._prior}

    @torch.inference_mode()
    def forward(self, x):
        with self._lock:
            if self.training or self._prior is not None:
                raise RuntimeError('Use one inference call at a time in eval mode')
            normalized = x/(x.abs().max(dim=-1, keepdim=True).values+1e-7)
            filtered = sosfiltfilt(self.sos, normalized.cpu().numpy().astype(np.float64),
                                  axis=-1).astype(np.float32)
            prior_wave = torch.from_numpy(np.ascontiguousarray(filtered)).to(x)
            self._prior = self.enhancer.normalizer(self.enhancer.to_mel(prior_wave), update=False)
            try:
                return self.enhancer(x)
            finally:
                self._prior = None


def load_filtered_reconstructor():
    for path, expected in PINNED_FILES.items():
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError('Missing or changed pinned reconstruction source or configuration')
    model, inference = reference.load_reconstructor('standard', boundary_safe=True)
    if model.lcfm.mode != model.lcfm.Mode.CFM or model.hp.force_gaussian_prior:
        raise RuntimeError('The filtered prior requires the pinned conditional-flow configuration')
    model.configurate_(nfe=32, solver='midpoint', lambd=.1, tau=.8)
    return FilteredPrior(model).eval(), inference


def process_file(model, inference, source, output, preset=PRESET, metadata=None):
    reference.validate_paths(source, output, metadata)
    if preset != PRESET:
        raise ValueError('Unknown filtered-prior preset')
    started = time.monotonic()
    audio, rate = sf.read(source, dtype='float32')
    result, gain = reference.reconstruct(model, inference, audio, rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        sf.write(handle, result, rate, format='WAV', subtype='FLOAT')
    report = {'preset': PRESET, 'settings': SETTINGS, 'seed': 2035,
              'source_commit': SOURCE_COMMIT, 'checkpoint_sha256': reference.CHECKPOINT_SHA256,
              'sample_rate': rate, 'samples': len(result),
              'channels': 1 if result.ndim == 1 else result.shape[1], 'gain': gain,
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
            parser.error('Input and output paths are required')
        try:
            reference.validate_paths(args.input, args.output, args.metadata)
        except ValueError as error:
            parser.error(str(error))
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        model, inference = load_filtered_reconstructor()
        if args.worker:
            print(json.dumps({'ready': True}), file=protocol, flush=True)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    meta = Path(request['metadata']) if request.get('metadata') else None
                    report = process_file(model, inference, Path(request['input']),
                                          Path(request['output']), request['preset'], meta)
                    reply = {'report': report}
                except Exception as error:
                    reply = {'error': f'{type(error).__name__}: {error}'}
                print(json.dumps(reply), file=protocol, flush=True)
            return
        if args.check:
            report = {'ready': True, 'preset': PRESET, 'settings': SETTINGS,
                      'checkpoint_sha256': reference.CHECKPOINT_SHA256}
        else:
            report = process_file(model, inference, args.input, args.output, metadata=args.metadata)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
