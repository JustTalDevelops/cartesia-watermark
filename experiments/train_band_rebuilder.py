"""Learn missing-band reconstruction from unmarked speech alone.

The network never receives its input's 1.5–3.5 kHz band. Targets are the same
unmarked recording, so this supplies paired reconstruction supervision without
assuming that two different TTS renders are waveform-aligned.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,resample
from band_rebuilder import BandRebuilder
from third_party.torch_pesq import PesqLoss
from third_party.stoi_loss import NegSTOILoss
from eval_remover import speech_metrics


def crop(x,rng=None):
    n=48000
    if len(x)<n:x=np.pad(x,(0,n-len(x)))
    start=rng.randrange(len(x)-n+1) if rng else (len(x)-n)//2
    return x[start:start+n]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--tag',default='r1')
    p.add_argument('--low',type=float,default=1500)
    p.add_argument('--high',type=float,default=3500)
    p.add_argument('--noise-input',action='store_true')
    p.add_argument('--waveform-weight',type=float,default=10.)
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.set_flush_denormal(True);torch.manual_seed(2044)
    rng=random.Random(2044)
    data=json.loads((ROOT/'results/remover_wave2c.manifest.json').read_text())
    data={k:[r for r in data[k] if not r['marked']] for k in ('train','validation')}
    data['validation']=data['validation'][:16]
    text=json.dumps(data,indent=2);prefix=ROOT/f'results/band_rebuilder_{args.tag}'
    if prefix.with_suffix('.pt').exists():raise ValueError('Tag exists')
    prefix.with_suffix('.manifest.json').write_text(text)
    arrays={}
    for split,rows in data.items():
        arrays[split]=[]
        for row in rows:
            x,sr=sf.read(ROOT/row['path'])
            arrays[split].append(resample(x,sr,16000).astype(np.float32))
    model=BandRebuilder(low=args.low,high=args.high,noise_input=args.noise_input)
    pesq=PesqLoss(1,sample_rate=16000).eval().requires_grad_(False)
    estoi=NegSTOILoss(16000,extended=True,do_resample=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    history=[];best=-float('inf');started=time.time()
    print('Loaded',len(data['train']),len(data['validation']),'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    for step in range(args.steps+1):
        if step%200==0:
            records=[]
            with torch.no_grad():
                model.eval()
                for row,a in zip(data['validation'],arrays['validation']):
                    x=torch.from_numpy(crop(a))[None];y=model(x)
                    metrics=speech_metrics(x[0].numpy(),y[0].numpy(),16000)
                    records.append(dict(path=row['path'],**metrics))
            summary=dict(step=step,seconds=time.time()-started,pesq_median=float(np.median([r['pesq_wb'] for r in records])),
                         estoi_median=float(np.median([r['estoi'] for r in records])),pesq_min=float(min(r['pesq_wb'] for r in records)))
            print(json.dumps(summary),flush=True);history.append(summary)
            if summary['pesq_median']>best:
                best=summary['pesq_median']
                state=dict(format='speech-band-rebuilder-v1',model_config=model.config,state_dict=model.state_dict(),
                           step=step,validation=summary,manifest_sha256=hashlib.sha256(text.encode()).hexdigest())
                temporary=prefix.with_suffix('.tmp.pt');torch.save(state,temporary);temporary.replace(prefix.with_suffix('.pt'))
                prefix.with_suffix('.validation.json').write_text(json.dumps(records,indent=2))
            prefix.with_suffix('.history.json').write_text(json.dumps(history,indent=2))
        if step==args.steps:break
        model.train();ix=rng.choices(range(len(arrays['train'])),k=4)
        x=torch.tensor(np.stack([crop(arrays['train'][i],rng) for i in ix]));y=model(x)
        scale=x.square().mean(-1).clamp_min(1e-10)
        loss=args.waveform_weight*((y-x).square().mean(-1)/scale).mean()
        spectral=0
        for size in (256,512,1024):
            window=torch.hann_window(size)
            a=torch.stft(x,size,size//4,window=window,return_complex=True).abs()
            b=torch.stft(y,size,size//4,window=window,return_complex=True).abs()
            spectral=spectral+((a-b).abs().mean((1,2))/a.mean((1,2)).clamp_min(1e-6)).mean()
        loss=loss+10*spectral/3+10*F.relu(4.4-pesq.mos(x,y)).mean()+100*F.relu(.995+estoi(y,x)).mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%100==99:print('step',step+1,'loss',float(loss.detach()),'seconds',round(time.time()-started),flush=True)


if __name__=='__main__':main()
