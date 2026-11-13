"""Adapt the waveform front end of XLS-R using actual Cartesia responses.

Unlike head-only distillation, gradients update the convolutional audio encoder.
Transformer blocks stay frozen. Reuses the raw oracle's voice-disjoint API data.
"""
import os
os.environ['FEAT'] = 'xlsr2'
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_remover import Teacher
from remover import ROOT
from train_raw_oracle import dataset, audio


@torch.no_grad()
def evaluate(model,rows,arrays):
    model.eval()
    scores = [float(model(torch.from_numpy(a)[None])) for a in arrays]
    y = np.array([r['label'] for r in rows],float); s = np.array(scores)
    summary = dict(clips=len(y), bce=float(np.mean(np.logaddexp(0,s)-y*s)),
                   accuracy=float(np.mean((s>0)==y)),auc=float(roc_auc_score(y,s)))
    for kind in sorted({r['kind'] for r in rows}):
        mask = np.array([r['kind']==kind for r in rows])
        summary[kind] = dict(clips=int(mask.sum()),accuracy=float(np.mean((s[mask]>0)==y[mask])))
    return summary,[dict(path=r['path'],label=r['label'],logit=s) for r,s in zip(rows,scores)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--tag',default='r1')
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(2042)
    rng=random.Random(2042)
    prefix=ROOT/f'results/encoder_oracle_{args.tag}'
    if prefix.with_suffix('.pt').exists():
        raise ValueError('Tag exists')
    rows=dataset();train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    manifest=json.dumps(dict(train=train,validation=val),indent=2)
    prefix.with_suffix('.manifest.json').write_text(manifest)
    tr_audio=[audio(r) for r in train];va_audio=[audio(r) for r in val]
    model=Teacher(config=ROOT/'results/oracle_detector_r1.json')
    model.encoder.feature_extractor.requires_grad_(True)
    model.encoder.feature_projection.requires_grad_(True)
    model.heads.requires_grad_(True)
    front=list(model.encoder.feature_extractor.parameters())+list(model.encoder.feature_projection.parameters())
    optimizer=torch.optim.AdamW([dict(params=front,lr=1e-5),dict(params=model.heads.parameters(),lr=1e-4)],weight_decay=.01)
    print('Loaded',len(train),len(val),'trainable',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    baseline,_=evaluate(model,val,va_audio);print('Baseline',json.dumps(baseline),flush=True)
    started=time.time();history=[];best=float('inf')
    for epoch in range(args.epochs):
        order=rng.choices(range(len(train)),weights=[r['weight'] for r in train],k=len(train))
        losses=[]
        for step,i in enumerate(order):
            # No random cropping: actual labels refer to the complete submitted clip.
            x=torch.from_numpy(tr_audio[i])[None]
            y=torch.tensor([float(train[i]['label'])])
            loss=F.binary_cross_entropy_with_logits(model(x),y)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            optimizer.step();losses.append(float(loss.detach()))
            if step%100==0:
                print('epoch',epoch+1,'clip',step,'loss',float(np.mean(losses[-50:])),flush=True)
        metrics,records=evaluate(model,val,va_audio)
        item=dict(epoch=epoch+1,seconds=time.time()-started,loss=float(np.mean(losses)),validation=metrics)
        history.append(item);print(json.dumps(item),flush=True)
        if metrics['bce']<best:
            best=metrics['bce']
            state=dict(format='cartesia-encoder-oracle-v1',encoder_state_dict=model.encoder.state_dict(),
                       heads_state_dict=model.heads.state_dict(),epoch=epoch+1,validation=metrics,
                       manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),training_args=vars(args))
            temporary=prefix.with_suffix('.tmp.pt');torch.save(state,temporary);temporary.replace(prefix.with_suffix('.pt'))
            prefix.with_suffix('.validation.json').write_text(json.dumps(records,indent=2))
            prefix.with_suffix('.json').write_text(json.dumps(dict(encoder_checkpoint=str(prefix.with_suffix('.pt').relative_to(ROOT)),
                feat='xlsr2',threshold=0.,training_voices=sorted({r['voice'] for r in train}),
                purpose='Training teacher with adapted audio front end; sealed4 excluded'),indent=2))
        prefix.with_suffix('.history.json').write_text(json.dumps(history,indent=2))


if __name__=='__main__':
    main()
