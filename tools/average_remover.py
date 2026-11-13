"""Average compatible trained weights, retaining the stricter edit budgets.

This is parameter interpolation, not output blending. Evaluate its actual
rendered audio before assuming it behaves like a blend of the two outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from remover import load_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--first-weight', type=float, default=.25)
    parser.add_argument('--selection-voice', action='append', default=[],
                        help='Voices used to choose this interpolation; excluded from future tests')
    args = parser.parse_args()
    if not 0 <= args.first_weight <= 1:
        parser.error('First weight must be between zero and one')
    manifest_path = args.output.with_suffix('.manifest.json')
    if args.output.exists() or manifest_path.exists():
        parser.error('Output exists; use a fresh name')
    models, checkpoints, manifests = [], [], []
    for path in (args.first, args.second):
        model, checkpoint = load_model(path)
        if checkpoint.get('format') != 'cartesia-remover-v1':
            parser.error('Only standard remover checkpoints can be averaged')
        source_manifest = path.with_suffix('.manifest.json')
        if digest(source_manifest) != checkpoint['manifest_sha256']:
            parser.error('Source manifest hash mismatch')
        models.append(model)
        checkpoints.append(checkpoint)
        manifests.append(json.loads(source_manifest.read_text()))
    config = dict(models[1].config)
    for key, first in models[0].config.items():
        second = config[key]
        if key in ('min_snr_db', 'local_snr_db'):
            values = [v for v in (first, second) if v is not None]
            config[key] = max(values) if values else None
        elif first != second:
            parser.error(f'Incompatible model setting: {key}')
    for split in ('train', 'validation'):
        if {r['voice'] for r in manifests[0][split]} != {r['voice'] for r in manifests[1][split]}:
            parser.error('Source voice splits differ; audit their lineage before averaging')
    states = [m.state_dict() for m in models]
    if states[0].keys() != states[1].keys():
        parser.error('Source state keys differ')
    state = {}
    for key, second in states[1].items():
        first = states[0][key]
        if first.shape != second.shape or first.dtype != second.dtype:
            parser.error(f'Incompatible tensor: {key}')
        state[key] = torch.lerp(second, first, args.first_weight)
    data = manifests[1]
    inherited = set(data.get('initialization_train_voices', []))
    calibration = set(args.selection_voice)
    for source, checkpoint in zip(manifests, checkpoints):
        inherited |= {r['voice'] for r in source['train']} | set(source.get('initialization_train_voices', []))
        calibration |= (set(source.get('oracle_calibration_voices', [])) |
                        set(checkpoint.get('teacher_config', {}).get('training_voices', [])) |
                        set(checkpoint.get('teacher_config', {}).get('calibration_voices', [])))
    data['initialization_train_voices'] = sorted(inherited)
    data['oracle_calibration_voices'] = sorted(calibration)
    data['weight_interpolation'] = {'first_weight': args.first_weight,
        'sources': [{'path': str(p), 'sha256': digest(p)} for p in (args.first, args.second)],
        'selection_voices': sorted(set(args.selection_voice))}
    if {r['voice'] for r in data['validation']} & (inherited | calibration):
        parser.error('Source or selection voices overlap validation')
    encoded = json.dumps(data, indent=2).encode()
    checkpoint = {'format': 'cartesia-remover-v1', 'state_dict': state, 'model_config': config,
                  'step': 0, 'manifest_sha256': hashlib.sha256(encoded).hexdigest(),
                  'teacher_config': checkpoints[1].get('teacher_config', {}),
                  'weight_interpolation': data['weight_interpolation'],
                  'limitations': 'Derived from trained weights; requires quality and endpoint verification'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    manifest_path.write_bytes(encoded)
    print(json.dumps({'checkpoint': str(args.output), 'sha256': digest(args.output),
                      'first_weight': args.first_weight, 'model_config': config}), flush=True)


if __name__ == '__main__':
    main()
