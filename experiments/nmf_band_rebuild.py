"""Fit a clean-speech spectral dictionary and infer the withheld marked band.

No detector loss or API requests. Every missing-band magnitude is predicted
using only observed frequencies and the unmarked training dictionary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal
from sklearn.decomposition import MiniBatchNMF
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,resample,quality_metrics
from eval_remover import speech_metrics


def transform(x):
    return torch.stft(torch.tensor(x,dtype=torch.float32)[None],1024,256,window=torch.hann_window(1024),return_complex=True)[0]


def reconstruct(z,components,whitening,low,high):
    freq=np.fft.rfftfreq(1024,1/24000)
    missing=(freq>=low)&(freq<=high)
    observed=~((freq>=low-100)&(freq<=high+100))
    v=z.abs().numpy()/whitening[:,None]
    a=components[:,observed].T.copy()
    gram=a.T@a
    numerator=a.T@v[observed]
    h=np.full(numerator.shape,.01,np.float32)
    for _ in range(160):h*=numerator/(gram@h+1e-8)
    predicted=(components.T@h)*whitening[:,None]
    target=z.abs().numpy().copy()
    # Only the excluded band uses predictions. Smooth shoulders avoid ringing.
    mask=np.minimum(np.clip((freq-low)/100,0,1),np.clip((high-freq)/100,0,1))[:,None]
    target=target*(1-mask)+predicted*mask
    rebuilt=torch.from_numpy(target)*torch.polar(torch.ones_like(z.real),z.angle())
    return rebuilt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--components',type=int,default=128)
    p.add_argument('--tag',default='r1')
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    manifest=json.loads((ROOT/'results/remover_wave2c.manifest.json').read_text())
    train=[r for r in manifest['train'] if not r['marked']]
    target=ROOT/f'results/nmf_band_{args.tag}.npz'
    if target.exists():
        data=np.load(target);components=data['components'];whitening=data['whitening']
    else:
        frames=[]
        for row in train:
            x,sr=sf.read(ROOT/row['path']);x=resample(x,sr,24000)
            scale=max(1e-5,np.sqrt(np.mean(x*x)))
            mag=transform(x*.1/scale).abs().numpy().T[::4]
            active=np.sum(mag*mag,axis=1)>np.max(np.sum(mag*mag,axis=1))*.001
            frames.append(mag[active])
        frames=np.concatenate(frames)
        whitening=np.maximum(frames.mean(0),1e-4).astype(np.float32)
        data=(frames/whitening).astype(np.float32)
        print('Fitting clean dictionary',data.shape,flush=True)
        model=MiniBatchNMF(n_components=args.components,init='nndsvda',random_state=2045,batch_size=1024,max_iter=80,tol=.001,max_no_improvement=15)
        model.fit(data);components=model.components_.astype(np.float32)
        np.savez(target,components=components,whitening=whitening)
        target.with_suffix('.manifest.json').write_text(json.dumps(dict(train=train,validation=manifest['validation'],
            source_manifest_sha256=hashlib.sha256((ROOT/'results/remover_wave2c.manifest.json').read_bytes()).hexdigest()),indent=2))
        print('Fit completed',model.n_iter_,flush=True)
    output=ROOT/f'audio/nmf_band_{args.tag}';output.mkdir(exist_ok=True)
    report=dict(model_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),clips=[])
    for name in ('es_228_sonic-preview','en_118_sonic-preview','ja_335_sonic-3.6','pt_346_sonic-3.6'):
        source=ROOT/'audio/corpus5'/f'{name}.wav';x,sr=sf.read(source)
        scale=max(1e-5,np.sqrt(np.mean(x*x)));x24=resample(x,sr,24000)*.1/scale
        z=transform(x24)
        for low,high in ((1500,3500),(1000,4000)):
            edited=reconstruct(z,components,whitening,low,high)
            y24=torch.istft(edited,1024,256,window=torch.hann_window(1024),length=len(x24)).numpy()*scale/.1
            delta=resample(y24-x24*scale/.1,24000,sr)[:len(x)]
            for strength in (.5,.75,1.):
                y=x+strength*delta
                row=dict(source=str(source.relative_to(ROOT)),low=low,high=high,strength=strength,
                         **quality_metrics(x,y),**speech_metrics(x,y,sr))
                if row['pesq_wb']>=4.1 and row['estoi']>=.975:
                    dest=output/f'{name}_{low}-{high}_{strength}.wav';sf.write(dest,y,sr,subtype='FLOAT');row['output']=str(dest.relative_to(ROOT))
                report['clips'].append(row);print(name[:2],low,high,strength,round(row['pesq_wb'],3),round(row['estoi'],4),flush=True)
    (ROOT/f'results/nmf_band_{args.tag}_probe.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
