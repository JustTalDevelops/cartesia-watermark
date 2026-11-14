"""Pinned local Vocos/EnCodec synthesis, loaded only in Python 3.12 workers."""
from importlib.metadata import version
import json
from math import gcd
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly

from codec_audio import digest, validate_audio

ROOT = Path(__file__).resolve().parent.parent


def assets():
    return json.loads((ROOT/'tools/codec_assets.json').read_text())


def verify_files(directory, expected):
    for name, checksum in expected.items():
        path = directory/name
        if not path.is_file() or digest(path) != checksum:
            raise RuntimeError(f'Install the pinned local codec assets before processing: {name}')


def check_environment():
    if (sys.version_info[:2] != (3, 12) or version('torch').split('+')[0] != '2.5.1' or
            version('transformers') != '4.46.3'):
        raise RuntimeError('Use the pinned Python 3.12 codec environment')


class NativeCodec:
    def __init__(self):
        check_environment()
        if version('encodec') != '0.1.1':
            raise RuntimeError('Install encodec==0.1.1 in the pinned codec environment')
        entry = assets()['codec']
        directory, source = ROOT/entry['local_directory'], ROOT/entry['source_directory']
        verify_files(directory, entry['files_sha256'])
        verify_files(source, entry['source_sha256'])
        import torch
        import yaml
        from encodec import EncodecModel
        configuration = yaml.safe_load((directory/'config.yaml').read_text())
        if ([configuration[k]['class_path'] for k in ('feature_extractor', 'backbone', 'head')] !=
                ['vocos.feature_extractors.EncodecFeatures', 'vocos.models.VocosBackbone', 'vocos.heads.ISTFTHead'] or
                configuration['feature_extractor']['init_args']['bandwidths'] != [1.5, 3., 6., 12.]):
            raise RuntimeError('Unexpected Vocos configuration')
        sys.path.insert(0, str(source))
        import vocos
        if Path(vocos.__file__).resolve().parent != (source/'vocos').resolve():
            raise RuntimeError('Vocos was imported from an unpinned installation')

        def local_state(name, repository=None):
            if name != 'encodec_24khz-d7cc33bc.th':
                raise RuntimeError('Unexpected EnCodec checkpoint request')
            return torch.load(directory/name, map_location='cpu', weights_only=True)

        torch.set_num_threads(4)
        previous = EncodecModel._get_pretrained
        try:
            EncodecModel._get_pretrained = staticmethod(local_state)
            model = vocos.Vocos.from_hparams(str(directory/'config.yaml'))
        finally:
            EncodecModel._get_pretrained = staticmethod(previous)
        state = torch.load(directory/'pytorch_model.bin', map_location='cpu', weights_only=True)
        state.update({'feature_extractor.encodec.'+name: value
                      for name, value in model.feature_extractor.encodec.state_dict().items()})
        model.load_state_dict(state, strict=True)
        self.model = model.eval().requires_grad_(False)
        self.metadata = {'model': entry['model_id'], 'revision': entry['revision'],
                         'source_commit': entry['source_commit'], 'files_sha256': entry['files_sha256'],
                         'sample_rate': 24000, 'bandwidth_kbps': 12., 'precision': 'float32',
                         'trained_by_this_project': False, 'model_downloads_during_processing': False}

    def reconstruct(self, audio, rate):
        import torch
        validate_audio(audio, rate)
        channels = audio[:, None] if audio.ndim == 1 else audio
        divisor, rebuilt = gcd(rate, 24000), []
        for channel in channels.T:
            native = resample_poly(channel, 24000//divisor, rate//divisor)
            padded = np.pad(native, (0, (-len(native)) % 320), mode='reflect')
            with torch.inference_mode():
                result = self.model(torch.from_numpy(padded)[None], bandwidth_id=torch.tensor([3])).flatten().numpy()
            if len(result) != len(padded) or not np.isfinite(result).all():
                raise RuntimeError('Codec returned invalid samples or the wrong length')
            restored = resample_poly(result[:len(native)], rate//divisor, 24000//divisor)[:len(audio)]
            if len(restored) != len(audio):
                raise RuntimeError('Codec did not preserve sample count')
            rebuilt.append(restored)
        result = np.stack(rebuilt, axis=1)
        result = result[:, 0] if audio.ndim == 1 else result
        gain = min(1., .999/max(1e-12, float(np.max(np.abs(result)))))
        return result*gain, gain
