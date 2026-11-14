"""Internal sigma0.5 BigVGAN plus complete codec-band reconstruction worker.

The input is a local manifest binding the original and native codec files. The
outer codec runtime performs content checks before publishing any recording.
"""
import argparse
from contextlib import redirect_stdout
import json
from math import gcd
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import resample_poly
import soundfile as sf

from codec_audio import digest, mix_reconstructions, quantize, validate_audio, validate_paths
from codec_settings import SETTINGS
from vocoder_inference import synthesize_chunks

ROOT = Path(__file__).resolve().parent.parent
PINNED = {'reconstruct_vocoder.py': '3378845287c3f1bf874147392184c5b15b43ca954adacd6eac484715e4564db0',
          'vocoder_inference.py': '38d02df4aa5d488aab7b37320dcbf6cc6e980178e23721783145adef138c37f2',
          'vocoder_assets.json': 'ea7870e32da6336a800a3ddbe555d527267124470b44f8e7e0c72603aae4e7f6'}


class BandRuntime:
    def __init__(self):
        from importlib.metadata import version
        if sys.version_info[:2] != (3, 13) or version('torch').split('+')[0] != '2.15.0':
            raise RuntimeError('Use the pinned Python 3.13 vocoder environment')
        for name, expected in PINNED.items():
            if digest(ROOT/'tools'/name) != expected:
                raise RuntimeError('The pinned base vocoder implementation changed')
        import reconstruct_vocoder
        self.model, self.features = reconstruct_vocoder.load_vocoder()
        self.assets = reconstruct_vocoder.ASSETS

    def reconstruct_base(self, audio, rate):
        import torch
        channels = audio[:, None] if audio.ndim == 1 else audio
        h, rebuilt = self.model.h, []
        divisor = gcd(rate, h.sampling_rate)
        with torch.inference_mode():
            for channel in channels.T:
                native = resample_poly(channel, h.sampling_rate//divisor, rate//divisor)
                padded = np.pad(native, (0, (-len(native)) % h.hop_size))
                mel = self.features(torch.as_tensor(padded)[None], h)
                mel = torch.from_numpy(gaussian_filter(mel.numpy(), sigma=(0, .5, .5), mode='reflect'))
                result = synthesize_chunks(mel, h.hop_size, lambda chunk: self.model(chunk).flatten().numpy())[:len(native)]
                restored = resample_poly(result, rate//divisor, h.sampling_rate//divisor)[:len(audio)]
                if len(restored) != len(audio) or not np.isfinite(restored).all():
                    raise RuntimeError('Vocoder changed the recording length or returned invalid samples')
                rebuilt.append(restored)
        result = np.stack(rebuilt, axis=1)
        result = result[:, 0] if audio.ndim == 1 else result
        gain = min(1., .999/max(1e-12, float(np.max(np.abs(result)))))
        return result*gain, gain

    def process_file(self, manifest_path, output, preset='codec-band', metadata=None):
        validate_paths(manifest_path, output, metadata)
        if preset != 'codec-band':
            raise ValueError('Unknown band reconstruction preset')
        manifest_sha = digest(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        source, native_path = Path(manifest['source']), Path(manifest['native'])
        if digest(source) != manifest['source_sha256'] or digest(native_path) != manifest['native_sha256']:
            raise RuntimeError('The original or codec input changed')
        original, rate = sf.read(source, dtype='float32')
        native, native_rate = sf.read(native_path, dtype='float32')
        validate_audio(original, rate)
        if native_rate != rate or native.shape != original.shape or not np.isfinite(native).all():
            raise RuntimeError('Codec input does not have the original complete sample grid')
        started = time.monotonic()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.codec-band-', dir=output.parent) as temporary:
            base, base_gain = self.reconstruct_base(original, rate)
            base = quantize(Path(temporary)/'vocoder.flac', base, rate)
            mixed, mix_gain = mix_reconstructions(base, native, rate)
            result = quantize(Path(temporary)/'mixed.flac', mixed, rate)
            assessed_flac_sha = digest(Path(temporary)/'mixed.flac')
        if (digest(manifest_path) != manifest_sha or digest(source) != manifest['source_sha256'] or
                digest(native_path) != manifest['native_sha256']):
            raise RuntimeError('A reconstruction input changed during synthesis')
        with output.open('xb') as handle:
            sf.write(handle, result, rate, format='WAV', subtype='FLOAT')
        report = {'preset': preset, 'source_sha256': manifest['source_sha256'], 'native_sha256': manifest['native_sha256'],
                  'output_sha256': digest(output), 'encoded_flac_sha256': assessed_flac_sha,
                  'settings': SETTINGS, 'model_id': self.assets['model_id'], 'model_revision': self.assets['model_revision'],
                  'checkpoint_sha256': self.assets['model_sha256']['bigvgan_generator.pt'],
                  'sample_rate': rate, 'samples': len(result), 'channels': 1 if result.ndim == 1 else result.shape[1],
                  'base_gain': base_gain, 'mix_gain': mix_gain, 'elapsed_seconds': time.monotonic()-started,
                  'quality_verified': False, 'content_assessment': 'Performed by the owning codec runtime'}
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
            parser.error('Input manifest and output WAV are required')
        validate_paths(args.input, args.output, args.metadata)
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        runtime = BandRuntime()
        if args.worker:
            print(json.dumps({'ready': True}), file=protocol, flush=True)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    report = runtime.process_file(Path(request['input']), Path(request['output']), request['preset'],
                                                  Path(request['metadata']) if request.get('metadata') else None)
                    reply = {'report': report}
                except Exception as error:
                    reply = {'error': f'{type(error).__name__}: {error}'}
                print(json.dumps(reply), file=protocol, flush=True)
            return
        report = {'ready': True} if args.check else runtime.process_file(args.input, args.output, metadata=args.metadata)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
