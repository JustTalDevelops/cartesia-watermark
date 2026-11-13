"""Adapt a raw-waveform detector using saved Cartesia endpoint responses.

No API calls. Keeps sealed4/h2h4 untouched and splits every derivative by its
source voice. Reproduces trailing padding used for historical API labelling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from remover import ROOT
from raw_oracle import RawOracle


def dataset():
    plans = {n: json.loads((ROOT/f'audio/corpus{n}_plan.json').read_text()) for n in (5, 6, 7)}
    voices3 = json.loads((ROOT/'audio/corpus3/voices.json').read_text())
    reserved = {r[2] for r in plans[7] if r[0] == 'sealed4'}
    validation_items = json.loads((ROOT/'audio/remover_validation/plan.json').read_text())['items']
    validation = {r['voice'] for r in validation_items}
    validation_by_name = {Path(r['path']).name:r['voice'] for r in validation_items}
    rows = []
    for row in json.loads((ROOT/'results/distill.json').read_text()).values():
        rows.append(dict(path=row['path'], label=row['detected'], voice=voices3[row['voice']],
                         padding=0, kind=row['stage'], weight=1.))
    for ledger, corpus, plan in (('val_cartesia', 'corpus5', plans[5]),
                                 ('sealed2_cartesia', 'sealed2', plans[5]),
                                 ('sealed3_cartesia', 'sealed3', plans[6])):
        for key, label in json.loads((ROOT/f'results/{ledger}.json').read_text()).items():
            name = Path(key).name
            actual_corpus = Path(key).parent.name if '/' in key else corpus
            item = plan[int(name.split('_')[1])]
            if item[0] != actual_corpus:
                raise ValueError(f'Plan mismatch: {key}')
            rows.append(dict(path=f'audio/{actual_corpus}/{name}', label=label, voice=item[2],
                             padding=5.2, kind='natural_api', weight=1.))
    for ledger in sorted((ROOT/'results').glob('remover_*_cartesia.json')):
        data = json.loads(ledger.read_text())
        for row in data.get('clips', []):
            if 'edited' not in row:
                continue
            source = Path(row['source'])
            voice = row.get('voice')
            if voice is None and source.parent.name == 'corpus5':
                voice = plans[5][int(source.name.split('_')[1])][2]
            if voice is None and source.parent.name == 'remover_validation':
                voice = validation_by_name[source.name]
            if voice is None:
                continue
            for path, label, digest in ((row['output'], row['edited'], row['output_sha256']),
                                         (row['source'], row.get('original'), row.get('source_sha256'))):
                if label is None:
                    continue
                rows.append(dict(path=path, label=label, voice=voice, padding=0,
                                 kind='remover_api', weight=3., expected_sha256=digest))
    # Deduplicate actual audio by its digest, and reject inconsistent labels.
    unique = {}
    for row in rows:
        if row['voice'] in reserved:
            continue
        path = ROOT/row['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get('expected_sha256') and digest != row['expected_sha256']:
            continue
        identity = (digest, row['padding'])
        if identity in unique:
            if unique[identity]['label'] != row['label']:
                raise ValueError('Conflicting endpoint labels on identical input')
            continue
        row['sha256'] = digest
        row['split'] = 'validation' if (row['voice'] in validation or int(hashlib.sha256(row['voice'].encode()).hexdigest()[:8],16)%8 == 0) else 'train'
        unique[identity] = row
    return list(unique.values())


def audio(row):
    x, sr = sf.read(ROOT/row['path'])
    if x.ndim == 2:
        x = x.mean(1)
    if row['padding'] and len(x) < row['padding']*sr:
        x = np.pad(x, (0, int(row['padding']*sr)-len(x)))
    divisor = np.gcd(sr, 16000)
    return resample_poly(x, 16000//divisor, sr//divisor).astype(np.float32)


def batch(rows, arrays, indices, augment=False, rng=None):
    items = [arrays[i] for i in indices]
    lengths = torch.tensor([len(x) for x in items])
    x = np.zeros((len(items), max(lengths).item()), np.float32)
    for j, item in enumerate(items):
        gain = 1.
        if augment:
            gain = 10**(rng.uniform(-3,3)/20) * rng.choice((-1,1))
        x[j, :len(item)] = item*gain
    return torch.from_numpy(x), lengths, torch.tensor([float(rows[i]['label']) for i in indices])


@torch.no_grad()
def evaluate(model, rows, arrays):
    model.eval()
    scores = [float(model(torch.from_numpy(x)[None])) for x in arrays]
    y = np.array([r['label'] for r in rows], float)
    s = np.array(scores)
    summary = dict(clips=len(y), bce=float(np.mean(np.logaddexp(0,s)-y*s)),
                   accuracy=float(np.mean((s>0)==y)), auc=float(roc_auc_score(y,s)))
    for kind in sorted({r['kind'] for r in rows}):
        mask = np.array([r['kind']==kind for r in rows])
        summary[kind] = dict(clips=int(mask.sum()), accuracy=float(np.mean((s[mask]>0)==y[mask])))
    return summary, [dict(path=r['path'], label=r['label'], logit=s) for r,s in zip(rows,scores)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--tag', default='r1')
    p.add_argument('--lr', type=float, default=1e-4)
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(2041)
    rng = random.Random(2041)
    prefix = ROOT/f'results/raw_oracle_{args.tag}'
    if prefix.with_suffix('.pt').exists():
        raise ValueError('Tag already exists')
    rows = dataset()
    train = [r for r in rows if r['split']=='train']
    val = [r for r in rows if r['split']=='validation']
    assert not ({r['voice'] for r in train} & {r['voice'] for r in val})
    manifest = json.dumps(dict(train=train, validation=val), indent=2)
    prefix.with_suffix('.manifest.json').write_text(manifest)
    tr_audio, va_audio = [audio(r) for r in train], [audio(r) for r in val]
    model = RawOracle()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    print('Loaded',len(train),len(val),'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    history, best, started = [], float('inf'), time.time()
    for epoch in range(args.epochs):
        # Weighted sampling includes the true-labelled adversarial edits often.
        order = rng.choices(range(len(train)), weights=[r['weight'] for r in train], k=len(train))
        losses = []
        model.train()
        for start in range(0,len(order),args.batch_size):
            ix = order[start:start+args.batch_size]
            x, lengths, y = batch(train,tr_audio,ix,True,rng)
            loss = F.binary_cross_entropy_with_logits(model(x,lengths),y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); losses.append(float(loss.detach()))
            if start%100==0:
                print('epoch',epoch+1,'clip',start,'loss',float(np.mean(losses[-50:])),flush=True)
        metrics, records = evaluate(model,val,va_audio)
        item = dict(epoch=epoch+1,seconds=time.time()-started,loss=float(np.mean(losses)),validation=metrics)
        history.append(item); print(json.dumps(item),flush=True)
        state = dict(format='cartesia-raw-oracle-v1',state_dict=model.state_dict(),epoch=epoch+1,
                     validation=metrics,manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
                     training_args=vars(args))
        if metrics['bce']<best:
            best=metrics['bce']; temporary=prefix.with_suffix('.tmp.pt')
            torch.save(state,temporary); temporary.replace(prefix.with_suffix('.pt'))
            prefix.with_suffix('.validation.json').write_text(json.dumps(records,indent=2))
        prefix.with_suffix('.history.json').write_text(json.dumps(history,indent=2))
        model.train()


if __name__ == '__main__':
    main()
