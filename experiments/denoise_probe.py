"""Test an independent pretrained speech denoiser on development audio.

Research dependency: facebookresearch/denoiser, CC BY-NC 4.0, checkout
8afd7c166699bb3c8b2d95b6dd706f71e1075df0 in .cache/denoiser.
Only its residual is converted back to the original rate. No API calls here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,resample,quality_metrics
from eval_remover import speech_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('master64','dns64','dns48'),default='master64')
    p.add_argument('--noise-snr',type=float,help='Purification: add reproducible white noise before denoising')
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    sys.path.insert(0,str(ROOT/'.cache/denoiser'))
    from denoiser import pretrained
    model=getattr(pretrained,args.model)().eval()
    tag = args.model + (f'_n{args.noise_snr:g}' if args.noise_snr is not None else '')
    out=ROOT/f'audio/denoise_{tag}';out.mkdir(exist_ok=True)
    report={'model':args.model,'upstream_revision':'8afd7c166699bb3c8b2d95b6dd706f71e1075df0','clips':[]}
    report.update(noise_snr=args.noise_snr,noise_seed=2043)
    rng=np.random.default_rng(2043)
    for name in ('es_228_sonic-preview','en_118_sonic-preview','ja_335_sonic-3.6','pt_346_sonic-3.6'):
        source=ROOT/'audio/corpus5'/f'{name}.wav';x,sr=sf.read(source)
        x16=resample(x,sr,16000).astype(np.float32)
        noisy=x16.copy()
        if args.noise_snr is not None:
            noise=rng.standard_normal(len(x16))
            noise*=np.sqrt(np.mean(x16**2))/10**(args.noise_snr/20)
            noisy+=noise.astype(np.float32)
        with torch.inference_mode():
            y16=model(torch.from_numpy(noisy)[None,None])[0,0].numpy().astype(float)
        delta=resample(y16-x16,16000,sr)[:len(x)]
        for band in ('full','1000-4000','1500-3500'):
            d=delta
            if band!='full':
                lo,hi=map(int,band.split('-'))
                d=signal.sosfiltfilt(signal.butter(6,(lo,hi),'bandpass',fs=sr,output='sos'),delta)
            for strength in (.5,.75,1.):
                y=x+strength*d
                row=dict(source=str(source.relative_to(ROOT)),band=band,strength=strength,
                         **quality_metrics(x,y),**speech_metrics(x,y,sr))
                if row['pesq_wb']>=4.1 and row['estoi']>=.98:
                    target=out/f'{name}_{band}_{strength}.wav'
                    sf.write(target,y,sr,subtype='FLOAT')
                    row.update(output=str(target.relative_to(ROOT)),output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                report['clips'].append(row)
                print(name[:2],band,strength,round(row['pesq_wb'],3),round(row['estoi'],4),flush=True)
    (ROOT/f'results/denoise_{tag}_probe.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
