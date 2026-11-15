"""Controlled TiCodec local/global quantization ablation with shared encoding.

The quantized control must retain the original adapter's exact output. Other
cells deliberately change the decoder inputs; none is a validated remover.
"""
from math import gcd
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'tools'))
from codec_audio import digest,validate_audio
from experiments.ticodec_adapter import TiCodec,ADAPTER_SHA256,ASSETS,ASSETS_SHA256

IMPLEMENTATION_SHA256=digest(Path(__file__))
MODES=(('quantized-control',True,True),('continuous-global',True,False),
       ('continuous-local',False,True),('continuous-both',False,False))
RATE,FRAME=24000,320


def reconstruct_modes(audio,rate,encode,decode,control_check=None):
    validate_audio(audio,rate)
    channels=audio[:,None] if audio.ndim==1 else audio
    divisor=gcd(rate,RATE);prepared=[]
    for channel in channels.T:
        native=np.asarray(resample_poly(channel,RATE//divisor,rate//divisor),dtype=np.float32)
        padded=np.pad(native,(0,(-len(native))%FRAME),mode='reflect')
        prepared.append((encode(padded),len(native),len(padded)))
    results={}
    for name,_,_ in MODES:
        rebuilt=[]
        for state,natural,padded in prepared:
            decoded=np.asarray(decode(state,name),dtype=np.float32)
            if decoded.ndim!=1 or len(decoded)!=padded or not np.isfinite(decoded).all():
                raise RuntimeError('Latent decoder returned malformed or incomplete native samples')
            restored=resample_poly(decoded[:natural],rate//divisor,RATE//divisor)[:len(audio)]
            if len(restored)!=len(audio) or not np.isfinite(restored).all():
                raise RuntimeError('Latent reconstruction changed the complete finite sample grid')
            rebuilt.append(restored)
        output=np.stack(rebuilt,axis=1)
        gain=min(1.,.999/max(1e-12,float(np.max(np.abs(output)))))
        output=np.asarray(output*gain,dtype=np.float32)
        output=output[:,0] if audio.ndim==1 else output
        results[name]=(output,gain)
        if name=='quantized-control' and control_check is not None:control_check(output,gain)
    return results


class TiCodecLatents:
    def __init__(self):
        self.base=TiCodec()
        self.metadata=dict(self.base.metadata)
        bitrate=self.metadata.pop('frame_codes_bitrate_bps')
        self.metadata.update(latent_ablation_sha256=IMPLEMENTATION_SHA256,
            baseline_frame_codes_bitrate_bps=bitrate,continuous_local_representation_bps=512*32*75,
            quantized_global_bits_per_complete_channel=80,continuous_global_bits_per_complete_channel=128*32,
            shared_encoder='One complete encode per channel for all four decoder cells',
            modes=[{'name':name,'local_quantized':local,'global_quantized':global_} for name,local,global_ in MODES],
            removal_efficacy='Unverified; spent-source development ablation')

    def prepare(self,native):
        import torch
        with torch.inference_mode():
            local,global_=self.base.encoder(torch.from_numpy(native)[None,None])
            _,_,local_codes,_,global_codes=self.base.quantizer(local,global_)
            local_codes=torch.stack([code.reshape(1,-1) for code in local_codes],-1)
            global_codes=torch.stack(global_codes,-1).unsqueeze(1)
            # Match the original adapter's exact code->embedding arithmetic.
            return {'local_raw':local,'global_raw':global_,
                    'local_quantized':self.base.quantizer.embed(local_codes),
                    'global_quantized':self.base.quantizer.embed_gst(global_codes).squeeze(-1)}

    def decode(self,state,mode):
        import torch
        policies={name:(local,global_) for name,local,global_ in MODES}
        if mode not in policies:raise ValueError('Use one of the four declared latent cells')
        local,global_=policies[mode]
        with torch.inference_mode():
            output=self.base.generator(state['local_quantized' if local else 'local_raw'],
                                       state['global_quantized' if global_ else 'global_raw'])
        return output[0,0].numpy()

    def reconstruct_modes(self,audio,rate,control_check=None):
        adapter=ROOT/'experiments/ticodec_adapter.py'
        if (digest(Path(__file__))!=IMPLEMENTATION_SHA256 or digest(adapter)!=ADAPTER_SHA256 or
                digest(ASSETS)!=ASSETS_SHA256 or
                any(digest(self.base.upstream/name)!=value for name,value in self.base.assets['source_files_sha256'].items())):
            raise ValueError('Loaded latent adapter, base adapter or pinned author code changed')
        return reconstruct_modes(audio,rate,self.prepare,self.decode,control_check)
