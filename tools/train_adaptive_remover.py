"""Unpaired restoration with an adapting speech-band discriminator.

Existing public/synthetic corpus only. No customer recordings, API calls, or
pre-watermark targets. Matching the unmarked-model distribution is a research
objective, not evidence of watermark removal. Native API verification is still
required after voice-fidelity evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from remover import ROOT, Remover, resample
from train_remover import fidelity
from third_party.stoi_loss import NegSTOILoss
from third_party.torch_pesq import PesqLoss


class Critic(nn.Module):
    def __init__(self, bandpass):
        super().__init__()
        self.register_buffer('bandpass', bandpass.detach().clone())
        self.nets = nn.ModuleList()
        for _ in range(3):
            layers = []
            for a, b, kernel, stride in [(1,16,15,4),(16,32,15,4),(32,64,15,4),
                                           (64,96,7,2),(96,1,3,1)]:
                layers.append(spectral_norm(nn.Conv1d(a,b,kernel,stride,padding=kernel//2)))
                if b != 1:
                    layers.append(nn.LeakyReLU(.2))
            self.nets.append(nn.Sequential(*layers))

    def forward(self, audio):
        scale = audio.square().mean(-1,keepdim=True).clamp_min(1e-10).sqrt()
        x = F.conv1d((audio/scale)[:,None],self.bandpass,padding=64)
        scores = []
        for index, net in enumerate(self.nets):
            view = F.avg_pool1d(x,2**index,2**index) if index else x
            scores.append(net(view).mean((1,2)))
        return torch.stack(scores).mean(0)


def discriminator_loss(clean_score, generated_score, marked_score):
    # Generated and untouched marked speech share the negative-class budget.
    # Equal total class weight avoids rewarding a constant negative answer.
    return (F.relu(1-clean_score)+.5*F.relu(1+generated_score)+.5*F.relu(1+marked_score)).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=800)
    parser.add_argument('--warmup',type=int,default=200)
    parser.add_argument('--validate-every',type=int,default=100)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--adversarial-weight',type=float,default=20.)
    parser.add_argument('--tag',default='adaptive24quality')
    args = parser.parse_args()
    if min(args.steps,args.warmup,args.validate_every,args.threads)<1:
        parser.error('Counts must be positive')
    if args.adversarial_weight <= 0:
        parser.error('Adversarial weight must be positive')
    if not args.tag.replace('_','').isalnum():
        parser.error('Use an alphanumeric experiment tag')
    torch.set_num_threads(args.threads);torch.set_flush_denormal(True)
    torch.manual_seed(2080);rng=np.random.default_rng(2080)
    out=ROOT/'results'/('remover_'+args.tag)
    if out.with_suffix('.pt').exists() or out.with_suffix('.manifest.json').exists():
        parser.error('Use a new tag; existing research artifacts are immutable')
    source=ROOT/'results/remover_native_generator_r2.manifest.json'
    data=json.loads(source.read_text())
    assert not {r['voice'] for r in data['train']} & {r['voice'] for r in data['validation']}
    reserved=json.loads((ROOT/'results/remover_test_batch3.plan.json').read_text())
    assert not {r['voice'] for s in ('train','validation') for r in data[s]} & {r['voice'] for r in reserved['items']}
    data.update(purpose='Adaptive discriminator unpaired speech-band restoration; no customer training or external requests',
                source_manifest=str(source.relative_to(ROOT)),source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                seed=2080,training_labels='Generator-model provenance, not exact clean waveform targets or new API labels')
    encoded=json.dumps(data,indent=2).encode();out.with_suffix('.manifest.json').write_bytes(encoded)
    size=4*16000;audio={};groups={}
    for split in ('train','validation'):
        groups[split]={}
        for row in data[split]:
            path=ROOT/row['path']
            if not path.resolve().is_relative_to((ROOT/'audio').resolve()):
                raise ValueError('Only existing corpus audio is allowed in training')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            x,sr=sf.read(path);x=x.mean(1) if x.ndim==2 else x
            x=resample(x,sr,16000).astype(np.float32)
            assert len(x)>=size and np.isfinite(x).all()
            audio[row['path']]=x
            groups[split].setdefault(row['voice'],{True:[],False:[]})[bool(row['marked'])].append(row['path'])
    pairs={s:[v for v,g in groups[s].items() if g[True] and g[False]] for s in groups}
    if not pairs['train']:
        raise ValueError('Same-voice marked/unmarked model pairs are required')

    def crop(path, random=True):
        x=audio[path]
        if random:
            starts=rng.integers(0,len(x)-size+1,size=8)
            start=max(starts,key=lambda s:float(np.mean(x[s:s+size]**2)))
        else:
            start=(len(x)-size)//2
        return torch.from_numpy(x[start:start+size].copy())

    def batch():
        marked,clean=[],[]
        for _ in range(2):
            voice=pairs['train'][int(rng.integers(len(pairs['train'])))];g=groups['train'][voice]
            marked.append(crop(g[True][int(rng.integers(len(g[True])))]))
            clean.append(crop(g[False][int(rng.integers(len(g[False])))]))
        return torch.stack(marked),torch.stack(clean)

    model=Remover(width=32,architecture='waveform',context_layers=7,min_snr_db=20.,local_snr_db=20.,
                  edit_low_hz=1000,edit_high_hz=4000,waveform_gain=.3)
    critic=Critic(model.bandpass_fir)
    opt_d=torch.optim.AdamW(critic.parameters(),lr=2e-4,betas=(.5,.9),weight_decay=1e-4)
    opt_g=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.5,.9),weight_decay=1e-5)
    started=time.monotonic()
    for step in range(1,args.warmup+1):
        marked,clean=batch();loss=(F.relu(1-critic(clean))+F.relu(1+critic(marked))).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite discriminator warmup')
        opt_d.zero_grad();loss.backward();opt_d.step()
        if step%50==0 or step==args.warmup:
            print('critic warmup',step,'loss',float(loss.detach()),'seconds',round(time.monotonic()-started),flush=True)
    monitor=copy.deepcopy(critic).eval().requires_grad_(False)
    perceptual=PesqLoss(1,sample_rate=16000).eval().requires_grad_(False)
    intelligibility=NegSTOILoss(16000,extended=True,do_resample=False).eval()
    val=[(row,crop(row['path'],False)) for row in data['validation']]

    @torch.no_grad()
    def validate():
        model.eval();critic.eval();records=[]
        for row,x in val:
            x=x[None];y=model(x);p=perceptual.mos(x,y);e=-intelligibility(y,x)
            records.append(dict(voice=row['voice'],marked=bool(row['marked']),fixed_monitor_score=float(monitor(y)[0]),
                                original_monitor_score=float(monitor(x)[0]),current_score=float(critic(y)[0]),
                                current_original_score=float(critic(x)[0]),proxy_pesq=float(p[0]),proxy_estoi=float(e[0])))
        pos=np.array([r['original_monitor_score'] for r in records if r['marked']]);neg=np.array([r['original_monitor_score'] for r in records if not r['marked']])
        # Higher score means the unmarked distribution. This fixed monitor is
        # withheld from generator gradients and is not Cartesia's detector.
        auc=float(((neg[:,None]>pos[None,:])+.5*(neg[:,None]==pos[None,:])).mean())
        def paired_auc(key):
            values=[]
            for voice in sorted({r['voice'] for r in records}):
                selected=[r for r in records if r['voice']==voice]
                positive=np.array([r[key] for r in selected if r['marked']])
                negative=np.array([r['current_original_score'] for r in selected if not r['marked']])
                if len(positive) and len(negative):
                    values.append(float(((negative[:,None]>positive[None,:])+.5*(negative[:,None]==positive[None,:])).mean()))
            if not values:raise ValueError('Validation has no same-voice provenance comparisons')
            return float(np.mean(values)),len(values)
        original_auc,voices=paired_auc('current_original_score')
        generated_auc,_=paired_auc('current_score')
        summary=dict(min_proxy_pesq=min(r['proxy_pesq'] for r in records),min_proxy_estoi=min(r['proxy_estoi'] for r in records),
                     fixed_monitor_marked_mean=float(np.mean([r['fixed_monitor_score'] for r in records if r['marked']])),
                     original_model_provenance_auc=auc,current_original_paired_auc=original_auc,
                     current_generated_paired_auc=generated_auc,paired_validation_voices=voices)
        model.train();critic.train();return summary,records

    def selection(summary):
        return (int(summary['min_proxy_pesq']<4.5)+int(summary['min_proxy_estoi']<.99),
                int(summary['current_original_paired_auc']<.65),
                max(0.,summary['current_generated_paired_auc']-.5),
                -summary['fixed_monitor_marked_mean'])

    def save(step,summary,suffix):
        payload=dict(format='cartesia-remover-v1',state_dict=model.state_dict(),model_config=model.config,
                     step=step,manifest_sha256=hashlib.sha256(encoded).hexdigest(),validation=summary,
                     training_args=vars(args),training_method='Unpaired adapting discriminator; fixed validation monitor excluded from gradients',
                     limitations='Research only. Provenance labels and GAN distribution matching do not prove watermark removal. Energy limits differ from V3; native fidelity and actual API checks required.',
                     trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if suffix=='.last.pt':
            payload.update(critic_state_dict=critic.state_dict(),monitor_state_dict=monitor.state_dict(),
                           generator_optimizer=opt_g.state_dict(),critic_optimizer=opt_d.state_dict())
        tmp=out.with_suffix(suffix+'.tmp');torch.save(payload,tmp);tmp.replace(out.with_suffix(suffix))

    summary,records=validate();best=selection(summary);save(0,summary,'.pt');save(0,summary,'.last.pt')
    history=[dict(step=0,**summary)];out.with_suffix('.initial.validation.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(history[-1]),flush=True)
    for step in range(1,args.steps+1):
        marked,clean=batch()
        with torch.no_grad():fake=model(marked)
        critic.requires_grad_(True)
        loss_d=discriminator_loss(critic(clean),critic(fake),critic(marked))
        opt_d.zero_grad();loss_d.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),5,error_if_nonfinite=True);opt_d.step()
        critic.requires_grad_(False)
        source_batch=torch.cat([marked,clean]);edited=model(source_batch)
        mse,spec=fidelity(source_batch,edited)
        p=perceptual.mos(source_batch,edited);e=-intelligibility(edited,source_batch)
        wave_weights=torch.tensor([100.,100.,500.,500.])
        loss_g=-args.adversarial_weight*critic(edited[:2]).mean()+(mse*wave_weights+20*spec).mean()+80*F.relu(4.58-p).mean()+400*F.relu(.995-e).mean()
        if not torch.isfinite(loss_d+loss_g):raise RuntimeError('Nonfinite adaptive restoration loss')
        opt_g.zero_grad();loss_g.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt_g.step()
        if step%25==0:
            print('step',step,'g_loss',round(float(loss_g.detach()),5),'d_loss',round(float(loss_d.detach()),5),'seconds',round(time.monotonic()-started),flush=True)
        if step%args.validate_every==0 or step==args.steps:
            summary,records=validate();history.append(dict(step=step,seconds=time.monotonic()-started,**summary))
            out.with_suffix('.history.json').write_text(json.dumps(history,indent=2));save(step,summary,'.last.pt')
            if selection(summary)<best:
                best=selection(summary);save(step,summary,'.pt');out.with_suffix('.validation.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(history[-1]),flush=True)
    print('ADAPTIVE_RESTORATION_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    main()
