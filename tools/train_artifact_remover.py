"""Paired artifact restoration on existing unmarked corpus audio.

Trains against known clean targets and synthetic band artifacts, without
watermark classifier feedback, external audio generation or customer uploads.
Transfer to Cartesia is an unverified research hypothesis until checked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from remover import ROOT, Remover, resample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--tag', default='paired23noise')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        parser.error('Use an alphanumeric tag')
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    torch.manual_seed(2077)
    rng = np.random.default_rng(2077)
    out = ROOT/'results'/('remover_'+args.tag)
    if out.with_suffix('.pt').exists():
        parser.error('Choose a new experiment tag')
    source = ROOT/'results/remover_native_generator_r2.manifest.json'
    source_data = json.loads(source.read_text())
    data = {split: [r for r in source_data[split] if not r['marked']] for split in ('train', 'validation')}
    assert not {r['voice'] for r in data['train']} & {r['voice'] for r in data['validation']}
    data.update(purpose='Paired restoration of synthetic speech-band artifacts; no classifier feedback or customer training audio',
                source_manifest=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), seed=2077)
    encoded = json.dumps(data, indent=2).encode()
    out.with_suffix('.manifest.json').write_bytes(encoded)
    audio = {}
    for split in ('train', 'validation'):
        audio[split] = []
        for row in data[split]:
            p = ROOT/row['path']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
            x, rate = sf.read(p)
            if x.ndim == 2:
                x = x.mean(1)
            audio[split].append(resample(x, rate, 16000).astype(np.float32))
    model = Remover(width=24, architecture='waveform', context_layers=5,
                    min_snr_db=30, local_snr_db=27, edit_low_hz=1000, edit_high_hz=4000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    size = 4*16000

    def batch(split, count, random):
        selected = random.integers(len(audio[split]), size=count)
        xs = []
        for index in selected:
            x = audio[split][index]
            start = int(random.integers(max(1, len(x)-size+1)))
            xs.append(np.pad(x[start:start+size], (0, max(0, size-len(x)))))
        return torch.from_numpy(np.stack(xs))

    def corrupt(clean, seed):
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(clean.shape, generator=generator)
        noise = F.conv1d(noise[:, None], model.bandpass_fir, padding=64)[:, 0]
        envelope = F.avg_pool1d(clean[:, None].square(), 321, 1, 160)[:, 0].clamp_min(1e-12).sqrt()
        noise *= envelope
        rms = clean.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        snr = 28 + 15*torch.rand((len(clean), 1), generator=generator)
        noise *= rms*10**(-snr/20)/noise.square().mean(-1, keepdim=True).clamp_min(1e-18).sqrt()
        # Include frequency-persistent, slowly varying patterns in addition
        # to independent noise, and identity examples to protect clean speech.
        time_axis = torch.arange(size)/16000
        for i in range(len(clean)):
            if i % 3 == 1:
                frequency = 1300 + 2400*torch.rand((), generator=generator)
                phase = 6.283185*torch.rand((), generator=generator)
                tone = torch.sin(6.283185*frequency*time_axis + phase)*envelope[i]
                noise[i] = tone*rms[i]*10**(-snr[i]/20)/tone.square().mean().clamp_min(1e-18).sqrt()
            if i % 3 == 2:
                noise[i].zero_()
        return clean+noise

    validation_clean = batch('validation', 8, np.random.default_rng(2078))
    validation_input = corrupt(validation_clean, 2078)

    def validate():
        with torch.inference_mode():
            restored = model(validation_input)
            untouched = model(validation_clean)
            denominator = validation_clean.square().mean(-1).clamp_min(1e-10)
            restoration = ((restored-validation_clean).square().mean(-1)/denominator).mean()
            identity = ((untouched-validation_clean).square().mean(-1)/denominator).mean()
            return {'objective': float(1000*(restoration+10*identity)),
                    'relative_restoration_error': float(restoration),
                    'relative_clean_edit_power': float(identity)}

    def save(step, summary, suffix):
        temporary = out.with_suffix(suffix+'.tmp')
        torch.save({'format':'cartesia-remover-v1', 'state_dict':model.state_dict(),
                    'model_config':model.config, 'step':step, 'validation':summary,
                    'manifest_sha256':hashlib.sha256(encoded).hexdigest(),
                    'training_args':vars(args), 'training_method':'paired synthetic artifact restoration',
                    'limitations':'Research model; no verified transfer to Cartesia watermark removal'}, temporary)
        temporary.replace(out.with_suffix(suffix))

    summary = validate()
    best = summary['objective']
    save(0, summary, '.pt')
    history = [{'step':0, **summary}]
    print(json.dumps(history[-1]), flush=True)
    started = time.monotonic()
    window = torch.hann_window(512)
    for step in range(1, args.steps+1):
        clean = batch('train', 3, rng)
        corrupted = corrupt(clean, 2077+step)
        restored = model(corrupted)
        denominator = clean.square().mean(-1).clamp_min(1e-10)
        error = (restored-clean).square().mean(-1)/denominator
        weights = torch.tensor([1., 1., 10.])
        loss = 1000*(error*weights).mean()
        spec_out = torch.stft(restored, 512, 128, window=window, return_complex=True)
        spec_ref = torch.stft(clean, 512, 128, window=window, return_complex=True)
        loss = loss+5*(torch.log1p(spec_out.abs())-torch.log1p(spec_ref.abs())).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite restoration loss')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            summary = validate()
            history.append({'step':step, 'seconds':time.monotonic()-started, **summary})
            out.with_suffix('.history.json').write_text(json.dumps(history, indent=2)+'\n')
            save(step, summary, '.last.pt')
            if summary['objective'] < best:
                best = summary['objective']
                save(step, summary, '.pt')
            print(json.dumps(history[-1]), flush=True)


if __name__ == '__main__':
    main()
