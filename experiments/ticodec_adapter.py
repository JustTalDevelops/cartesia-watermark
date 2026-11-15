"""Whole-channel CPU adapter for the pinned author TiCodec 1g4r checkpoint.

No downloads occur here. This is a research codec, not a validated remover.
The original model code stays in the separately obtained author checkout.
"""
import json
from math import gcd
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest, validate_audio

ASSETS = Path(__file__).with_name('ticodec_assets.json')
ADAPTER_SHA256 = digest(Path(__file__))
ASSETS_SHA256 = digest(ASSETS)
RATE, FRAME = 24000, 320


def reconstruct_channels(audio, rate, decode):
    """Reflect only to a model frame; restore the original grid with shared gain."""
    validate_audio(audio,rate)
    channels = audio[:,None] if audio.ndim==1 else audio
    restored = []
    divisor = gcd(rate,RATE)
    for channel in channels.T:
        native = np.asarray(resample_poly(channel,RATE//divisor,rate//divisor),dtype=np.float32)
        padded = np.pad(native,(0,(-len(native))%FRAME),mode='reflect')
        result = np.asarray(decode(padded),dtype=np.float32)
        if result.ndim != 1 or len(result) != len(padded) or not np.isfinite(result).all():
            raise RuntimeError('TiCodec must return one finite sample for every padded native sample')
        output = resample_poly(result[:len(native)],rate//divisor,RATE//divisor)[:len(audio)]
        if len(output) != len(audio) or not np.isfinite(output).all():
            raise RuntimeError('TiCodec did not preserve the complete finite sample grid')
        restored.append(output)
    output = np.stack(restored,axis=1)
    gain = min(1., .999/max(1e-12,float(np.max(np.abs(output)))))
    output = np.asarray(output*gain,dtype=np.float32)
    return (output[:,0] if audio.ndim==1 else output),gain


class TiCodec:
    def __init__(self):
        from codec_inference import check_environment
        check_environment()
        import torch
        torch.set_num_threads(4)
        assets = json.loads(ASSETS.read_text())
        upstream, checkpoint = ROOT/assets['local_code_directory'], ROOT/assets['local_weights_file']
        if (any(digest(upstream/name) != value for name,value in assets['source_files_sha256'].items()) or
                digest(checkpoint) != assets['weights_sha256'] or checkpoint.stat().st_size != assets['weights_bytes'] or
                digest(Path(__file__)) != ADAPTER_SHA256 or digest(ASSETS) != ASSETS_SHA256):
            raise ValueError('Use the unchanged pinned author code, configuration and checkpoint')
        sys.path.insert(0,str(upstream))
        from academicodec.models.ticodec import models
        from academicodec import utils
        from academicodec.models.ticodec import env
        from academicodec.models.ticodec.env import AttrDict
        if any(Path(module.__file__).resolve() != (upstream/name).resolve() for module,name in (
                (models,'academicodec/models/ticodec/models.py'),
                (utils,'academicodec/utils.py'),(env,'academicodec/models/ticodec/env.py'))):
            raise ValueError('A different installed TiCodec module shadowed the pinned author code')
        config = json.loads((upstream/assets['configuration_file']).read_text())
        h = AttrDict(config)
        self.encoder,self.generator,self.quantizer = models.Encoder(h),models.Generator(h),models.Quantizer(h)
        weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
        for name in ('encoder','generator','quantizer'):
            module = getattr(self,name)
            module.load_state_dict(weights[name],strict=True)
            module.eval().requires_grad_(False)
        self.assets,self.upstream = assets,upstream
        self.metadata = {k:assets[k] for k in ('model','trained_by_this_project','code_repository','code_revision',
            'code_license','weights_repository','weights_revision','weights_sha256','weights_license',
            'source_files_sha256','frame_codes_bitrate_bps','global_codes_per_complete_channel')}
        self.metadata.update(sample_rate=RATE,precision='float32',device='cpu',threads=4,
            adapter_sha256=ADAPTER_SHA256,assets_manifest_sha256=ASSETS_SHA256,model_downloads_during_processing=False,
            channel_policy='Complete channels independently; one global token set per complete channel',
            boundary_policy='Reflect to 320 samples, decode, discard model reflection, restore original grid',
            weight_normalization='Original author layers retained',removal_efficacy='Unverified')

    def decode(self,native):
        import torch
        if native.ndim!=1 or len(native)%FRAME or not np.isfinite(native).all():
            raise ValueError('Use a finite complete native waveform on the model frame grid')
        with torch.inference_mode():
            latent,style = self.encoder(torch.from_numpy(native)[None,None])
            _,_,local_codes,_,global_codes = self.quantizer(latent,style)
            local_codes = torch.stack([code.reshape(1,-1) for code in local_codes],-1)
            global_codes = torch.stack(global_codes,-1).unsqueeze(1)
            local = self.quantizer.embed(local_codes)
            global_style = self.quantizer.embed_gst(global_codes).squeeze(-1)
            result = self.generator(local,global_style)
        return result[0,0].numpy()

    def reconstruct(self,audio,rate):
        if (digest(Path(__file__))!=ADAPTER_SHA256 or digest(ASSETS)!=ASSETS_SHA256 or
                any(digest(self.upstream/name)!=value for name,value in self.assets['source_files_sha256'].items())):
            raise ValueError('Loaded codec adapter or pinned author code changed')
        return reconstruct_channels(audio,rate,self.decode)
