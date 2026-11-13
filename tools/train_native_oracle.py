"""Adapt the waveform oracle on native-length actual API labels.

High-quality edited counterexamples receive extra sampling weight. All reserved
voices and all native development voices remain outside oracle training.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from raw_oracle import load_raw_oracle
from remover import ROOT, resample


def same_source_groups(rows):
    groups = defaultdict(lambda: {False: [], True: []})
    for index, row in enumerate(rows):
        for source in row.get('reference_sources', []):
            groups[source][bool(row['label'])].append(index)
    result = []
    for source, labels in groups.items():
        if labels[False] and labels[True]:
            voices = {rows[i]['voice'] for indices in labels.values() for i in indices}
            if len(voices) != 1:
                raise ValueError('Same-source pair contains different voices')
            result.append((source, labels[True], labels[False], next(iter(voices))))
    return result


def paired_metrics(records, high_quality=False, margin=.5):
    per_source, pairs = [], 0
    for _, positive, negative, _ in same_source_groups(records):
        differences = []
        for p in positive:
            for n in negative:
                a, b = records[p], records[n]
                if high_quality and (any(r['kind'] not in ('natural', 'high_quality_edit') for r in (a, b)) or
                                     all(r['kind'] != 'high_quality_edit' for r in (a, b))):
                    continue
                differences.append(a['logit']-b['logit'])
        if differences:
            values = np.asarray(differences)
            pairs += len(values)
            per_source.append((float(np.mean(values > 0)), float(np.mean(np.logaddexp(0, margin-values)))))
    return {'utterances': len(per_source), 'pairs': pairs,
            'accuracy': float(np.mean([r[0] for r in per_source])) if per_source else None,
            'loss': float(np.mean([r[1] for r in per_source])) if per_source else None}


class EncoderOracle(torch.nn.Module):
    """The same XLS-R features and pooling used by the remover teacher."""
    def __init__(self, checkpoint):
        super().__init__()
        os.environ['FEAT'] = 'xlsr2'
        import frames
        import train_head as th
        self.encoder = frames.model()
        self.heads = torch.nn.ModuleList([th.Conv()])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.heads.load_state_dict(checkpoint['heads_state_dict'])
        self.requires_grad_(False)
        self.encoder.feature_extractor.requires_grad_(True)
        self.heads.requires_grad_(True)

    def forward(self, x, lengths=None):
        from train_remover import Teacher
        if lengths is not None and not bool((lengths == x.shape[-1]).all()):
            raise ValueError('Encoder batches must contain equal native crop lengths')
        z = (x-x.mean(-1, keepdim=True))/(x.std(-1, correction=0, keepdim=True)+1e-9)
        h = self.encoder(z, output_hidden_states=True).hidden_states[2]
        return Teacher.pool(h, x, self.heads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'results/api_oracle_native_r2.manifest.json')
    parser.add_argument('--architecture', choices=('raw', 'encoder'), default='raw')
    parser.add_argument('--init', type=Path)
    parser.add_argument('--initial-manifest', type=Path,
                        help='Audited parent manifest, including inherited training voices')
    parser.add_argument('--initial-lineage-config', type=Path,
                        help='Hash-bound ancestry metadata for older checkpoints missing inherited voices')
    parser.add_argument('--tag', default='native_r2')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--paired-fraction', type=float, default=0., help='Probability of a synchronized actual-positive/negative same-source batch')
    parser.add_argument('--ranking-weight', type=float, default=0.)
    parser.add_argument('--ranking-margin', type=float, default=.5)
    args = parser.parse_args()
    if not 0 <= args.paired_fraction <= 1 or min(args.ranking_weight, args.ranking_margin) < 0:
        parser.error('Invalid paired supervision settings')
    if args.paired_fraction and args.batch_size != 2:
        parser.error('Paired supervision requires batch size two')
    if args.init is None:
        args.init = ROOT/('results/raw_oracle_r1.pt' if args.architecture == 'raw'
                         else 'results/encoder_oracle_r1_epoch1.pt')
    if args.initial_manifest is None:
        args.initial_manifest = ROOT/('results/raw_oracle_r1.manifest.json' if args.architecture == 'raw'
                                     else 'results/encoder_oracle_r1.manifest.json')
    if min(args.steps, args.batch_size, args.validate_every, args.threads) < 1 or args.lr <= 0:
        parser.error('Training counts and learning rate must be positive')
    if not args.tag.replace('_', '').replace('-', '').isalnum():
        parser.error('Use an alphanumeric tag')
    torch.set_num_threads(args.threads)
    torch.manual_seed(2034)
    rng = np.random.default_rng(2034)
    prefix = ROOT/f'results/{args.architecture}_oracle_{args.tag}'
    if prefix.with_suffix('.pt').exists():
        parser.error('Checkpoint exists; use a new tag')
    raw_manifest = args.manifest.read_bytes()
    data = json.loads(raw_manifest)
    prefix.with_suffix('.manifest.json').write_bytes(raw_manifest)
    initial = torch.load(args.init, map_location='cpu', weights_only=True)
    parent_bytes = args.initial_manifest.read_bytes()
    if initial['manifest_sha256'] != hashlib.sha256(parent_bytes).hexdigest():
        raise ValueError('Parent training manifest does not match initialization checkpoint')
    parent = json.loads(parent_bytes)
    inherited = (set(initial.get('training_voices', [])) |
                 {r['voice'] for r in parent['train']} |
                 set(parent.get('initialization_train_voices', [])) |
                 set(parent.get('oracle_calibration_voices', [])))
    if args.initial_lineage_config:
        lineage = json.loads(args.initial_lineage_config.read_text())
        if lineage.get('checkpoint_sha256') != hashlib.sha256(args.init.read_bytes()).hexdigest():
            raise ValueError('Ancestry config does not belong to initialization checkpoint')
        inherited |= set(lineage.get('training_voices', [])) | set(lineage.get('calibration_voices', []))
    training_voices = inherited | {r['voice'] for r in data['train']}
    validation_voices = {r['voice'] for r in data['validation']}
    if training_voices & validation_voices:
        raise ValueError('Initialization or current training voices overlap validation')
    if args.architecture == 'raw':
        model, initial = load_raw_oracle(args.init)
        model.requires_grad_(True)
        head = list(model.detector[-1].parameters())
    else:
        if initial.get('format') != 'cartesia-encoder-oracle-v1':
            raise ValueError('Unsupported encoder initialization checkpoint')
        model = EncoderOracle(initial)
        head = list(model.heads.parameters())
    # train_head's legacy import sets global threads/RNG; restore this run's
    # controls after construction and before sampling/dropout training.
    torch.set_num_threads(args.threads)
    torch.manual_seed(2034)
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW([{'params': body, 'lr': args.lr},
                                   {'params': head, 'lr': args.lr*5}], weight_decay=1e-4)
    audio = {}
    for split in ('train', 'validation'):
        for row in data[split]:
            path = ROOT/row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Labeled source changed')
            x, rate = sf.read(path)
            if len(x)/rate < 5 or row.get('padding', 0):
                raise ValueError('Oracle training requires native duration >=5 seconds')
            x = x.mean(1) if x.ndim == 2 else x
            audio[row['sha256']] = resample(x, rate, 16000).astype(np.float32)
    voice_weight = defaultdict(float)
    for row in data['train']:
        voice_weight[row['voice']] += row['weight']
    probability = np.array([r['weight']/voice_weight[r['voice']] for r in data['train']])
    label_weight = {label: sum(p for p, r in zip(probability, data['train']) if r['label'] == label)
                    for label in (False, True)}
    probability /= np.array([label_weight[r['label']] for r in data['train']])
    probability /= probability.sum()
    pair_groups = same_source_groups(data['train'])
    if args.paired_fraction and not pair_groups:
        parser.error('No same-source actual positive/negative pairs')
    pair_voice_count = Counter(group[3] for group in pair_groups)
    pair_probability = np.array([max(data['train'][i]['weight'] for i in p+n)/pair_voice_count[voice]
                                 for _, p, n, voice in pair_groups])
    if pair_groups:
        pair_probability /= pair_probability.sum()

    @torch.no_grad()
    def validate():
        model.eval()
        records = []
        for row in data['validation']:
            x = torch.from_numpy(audio[row['sha256']])[None]
            logit = float(model(x)[0])
            records.append({'path': row['path'], 'label': row['label'], 'kind': row['kind'],
                            'voice': row['voice'], 'logit': logit,
                            'reference_sources': row.get('reference_sources', []),
                            'bce': float(np.logaddexp(0, logit)-row['label']*logit)})
        def metrics(rows):
            labels = [r['label'] for r in rows]
            grouped = [[r for r in rows if r['label'] == value] for value in (False, True)]
            return {'clips': len(rows),
                    'balanced_bce': float(np.mean([np.mean([r['bce'] for r in group]) for group in grouped if group])),
                    'accuracy': float(np.mean([(r['logit'] > 0) == r['label'] for r in rows])),
                    'balanced_accuracy': float(np.mean([np.mean([(r['logit'] > 0) == r['label'] for r in group]) for group in grouped if group])),
                    'auc': float(roc_auc_score(labels, [r['logit'] for r in rows])) if len(set(labels)) == 2 else None}
        all_metrics = metrics(records)
        high = metrics([r for r in records if r['kind'] == 'high_quality_edit'])
        paired_high = paired_metrics(records, high_quality=True, margin=args.ranking_margin)
        return {'all': all_metrics, 'high_quality_edits': high,
                'same_source_pairs': paired_metrics(records, margin=args.ranking_margin),
                'high_quality_pairs': paired_high,
                'objective': high['balanced_bce'] + .25*all_metrics['balanced_bce'] +
                             args.ranking_weight*(paired_high['loss'] or 0.)}, records

    summary, records = validate()
    print(json.dumps({'step': 0, **summary}), flush=True)
    best = float('inf')
    history = [{'step': 0, **summary}]
    started = time.time()
    running = []
    for step in range(0, args.steps+1):
        # Keep the frozen transformer deterministic during frontend adaptation.
        model.train() if args.architecture == 'raw' else model.eval()
        if args.architecture == 'encoder':
            model.heads.train()
        if step == 0:
            indices = []
            paired = False
        else:
            paired = bool(args.paired_fraction and rng.random() < args.paired_fraction)
            if paired:
                _, positive, negative, _ = pair_groups[int(rng.choice(len(pair_groups), p=pair_probability))]
                indices = []
                for pool in (positive, negative):
                    weights = np.array([data['train'][i]['weight'] for i in pool]); weights /= weights.sum()
                    indices.append(int(rng.choice(pool, p=weights)))
            else:
                indices = rng.choice(len(data['train']), size=args.batch_size, p=probability)
        if args.architecture == 'encoder':
            # Independent dropout masks would swamp the tiny waveform changes
            # in a synchronized positive/negative pair. Unpaired batches retain
            # the original head dropout regularization.
            model.heads.train(not paired)
        rows = [data['train'][int(i)] for i in indices]
        chunks = []
        if paired:
            common = min(len(audio[row['sha256']]) for row in rows)
            pair_length = min(common, 96000) if args.architecture == 'raw' else 80000
            pair_start = int(rng.integers(0, common-pair_length+1))
        for row in rows:
            x = audio[row['sha256']]
            length = pair_length if paired else (min(len(x), 96000) if args.architecture == 'raw' else 80000)
            start = pair_start if paired else int(rng.integers(0, len(x)-length+1))
            chunk = x[start:start+length].copy()
            if not paired and rng.random() < .25:
                noise_scale = np.sqrt(np.mean(chunk**2))*10**(-rng.uniform(40, 55)/20)
                chunk += rng.standard_normal(length).astype(np.float32)*noise_scale
            chunks.append(chunk)
        if paired and rng.random() < .25:
            scale = np.sqrt(np.mean(np.square(chunks)))*10**(-rng.uniform(40, 55)/20)
            noise = rng.standard_normal(pair_length).astype(np.float32)*scale
            chunks = [chunk+noise for chunk in chunks]
        if step:
            lengths = torch.tensor([len(x) for x in chunks])
            batch = torch.from_numpy(np.stack([np.pad(x, (0, int(lengths.max())-len(x))) for x in chunks]))
            labels = torch.tensor([r['label'] for r in rows], dtype=torch.float32)
            logits = model(batch, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if paired and args.ranking_weight:
                loss = loss + args.ranking_weight*F.softplus(args.ranking_margin-(logits[0]-logits[1]))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite oracle loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            running.append(float(loss.detach()))
        if step and step % 20 == 0:
            loss_name = 'loss' if args.ranking_weight else 'bce'
            print(f'step {step}/{args.steps} {loss_name}={np.mean(running[-20:]):.4f} elapsed={time.time()-started:.0f}s', flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            if step:
                summary, records = validate()
            row = {'step': step, 'seconds': time.time()-started, **summary}
            if step:
                history.append(row)
            prefix.with_suffix('.history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps(row), flush=True)
            state = {'step': step, 'validation': summary, 'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
                     'runtime_threads': torch.get_num_threads(), 'runtime_torch_seed': torch.initial_seed(),
                     'source_checkpoint': str(args.init), 'source_checkpoint_sha256': hashlib.sha256(args.init.read_bytes()).hexdigest(),
                     'training_args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     'training_voices': sorted(training_voices),
                     'current_training_voices': sorted({r['voice'] for r in data['train']}),
                     'inherited_training_voices': sorted(inherited),
                     'validation_voices': sorted(validation_voices)}
            if args.architecture == 'raw':
                state.update(format='cartesia-raw-oracle-v1', state_dict=model.state_dict())
            else:
                state.update(format='cartesia-encoder-oracle-v1',
                             encoder_state_dict=model.encoder.state_dict(),
                             heads_state_dict=model.heads.state_dict())
            temp = prefix.with_suffix('.last.pt.tmp')
            torch.save(state, temp)
            temp.replace(prefix.with_suffix('.last.pt'))
            if summary['objective'] < best:
                best = summary['objective']
                temp = prefix.with_suffix('.pt.tmp')
                torch.save(state, temp)
                temp.replace(prefix.with_suffix('.pt'))
                prefix.with_suffix('.validation.json').write_text(json.dumps(records, indent=2))
                config = {('raw_checkpoint' if args.architecture == 'raw' else 'encoder_checkpoint'):
                          str(prefix.with_suffix('.pt').relative_to(ROOT)),
                          'threshold': 0., 'training_voices': sorted(training_voices),
                          'current_training_voices': state['current_training_voices'],
                          'inherited_training_voices': sorted(inherited),
                          'validation_voices': sorted(validation_voices),
                          'checkpoint_sha256': hashlib.sha256(prefix.with_suffix('.pt').read_bytes()).hexdigest(),
                          'validation': summary}
                prefix.with_suffix('.json').write_text(json.dumps(config, indent=2))
                print('Saved improved native oracle', flush=True)


if __name__ == '__main__':
    main()
