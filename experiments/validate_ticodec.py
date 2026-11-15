"""Compare the CPU adapter with the author's encode/decode API on synthetic audio.

This loads real weights. It measures implementation parity, never removal.
Run only when no other neural generation/scoring process is active.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from experiments.ticodec_adapter import TiCodec,ADAPTER_SHA256,ASSETS,ASSETS_SHA256
from experiments.probe_codec_frames import without_http,save
from codec_audio import digest


def run(directory):
    if directory.exists():raise ValueError('Choose a new parity directory')
    directory.mkdir(parents=True)
    path=directory/'parity.json'
    report={'kind':'TiCodec CPU synthetic encode/decode parity','complete':False,
        'adapter_sha256':ADAPTER_SHA256,'assets_manifest_sha256':ASSETS_SHA256,
        'validation_code_sha256':digest(Path(__file__)), 'new_http_requests':0,'new_tts_requests':0,
        'fresh_efficacy_sources':0,'speech_quality_evaluated':False,'detector_checked':False}
    save(path,report)
    try:
        with without_http():
            import numpy as np
            import torch
            started=time.perf_counter()
            model=TiCodec()
            report.update(model=model.metadata,model_load_seconds=time.perf_counter()-started)
            save(path,report)
            from academicodec.models.ticodec.vqvae import VQVAE
            # Construct the reference API around the same strictly loaded weights.
            # Its constructor's unrestricted torch.load is never called.
            reference=VQVAE.__new__(VQVAE)
            torch.nn.Module.__init__(reference)
            reference.encoder,reference.generator,reference.quantizer=model.encoder,model.generator,model.quantizer
            reference.eval().requires_grad_(False)
            t=np.arange(120000,dtype=np.float32)/24000
            native=np.asarray(.12*np.sin(2*np.pi*(170*t+35*t*t))+.03*np.sin(2*np.pi*2201*t),dtype=np.float32)
            started=time.perf_counter()
            adapted=model.decode(native)
            report['adapter_seconds']=time.perf_counter()-started
            started=time.perf_counter()
            with torch.inference_mode():
                codes,global_codes=reference.encode(torch.from_numpy(native)[None])
                original=reference(codes,global_codes)[0,0].numpy()
            report.update(reference_seconds=time.perf_counter()-started,native_samples=len(native),
                frame_codes_shape=list(codes.shape),global_codes_shape=list(global_codes.shape),
                author_api_device='cpu',author_weight_normalization='retained',
                shared_loaded_weights=True,maximum_absolute_error=float(np.max(np.abs(adapted-original))),
                exact_author_api_waveform=np.array_equal(adapted,original),
                original_sample_count_preserved=adapted.shape==native.shape,
                finite_samples=bool(np.isfinite(adapted).all()))
            if not all(report[k] for k in ('exact_author_api_waveform','original_sample_count_preserved','finite_samples')):
                raise ValueError('Adapter differs from the pinned author encode/decode API')
            if digest(ASSETS)!=ASSETS_SHA256 or digest(Path(__file__))!=report['validation_code_sha256']:
                raise ValueError('Validation code or asset registry changed')
            report['complete']=True
            save(path,report)
            print('TICODEC_SYNTHETIC_PARITY',report['exact_author_api_waveform'],flush=True)
    except BaseException as error:
        report['error_type']=type(error).__name__;save(path,report);raise
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    run(parser.parse_args().directory)
