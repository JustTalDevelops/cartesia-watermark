"""Export an immutable inference checkpoint and its audited voice manifest.

    .venv/bin/python tools/export_remover.py results/remover_run.pt results/remover_quality_v2.pt

Evaluation and API reports should reference the exported file's hash. Keep
later measurements in a separate release report so the tested model never changes.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from remover import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest_path = args.source.with_suffix('.manifest.json')
    output_manifest = args.output.with_suffix('.manifest.json')
    if args.output.suffix != '.pt':
        parser.error('Use a .pt checkpoint path')
    if args.output.exists() or output_manifest.exists():
        parser.error('Export destination already exists; use a new version')
    raw = args.source.read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    manifest = manifest_path.read_bytes()
    if checkpoint['format'] not in ('cartesia-remover-v1', 'cartesia-quality-blend-v1'):
        parser.error('Expected a residual editor or quality blend')
    if hashlib.sha256(manifest).hexdigest() != checkpoint['manifest_sha256']:
        parser.error('Source manifest hash differs from checkpoint')
    model, _ = load_model(args.source)
    keep = ('format', 'state_dict', 'step', 'manifest_sha256', 'training_args',
            'teacher_config', 'teacher_member_sha256', 'validation', 'limitations',
            'weight_interpolation', 'quality_profile_selection', 'component_provenance')
    exported = {key: checkpoint[key] for key in keep if key in checkpoint}
    exported['model_config'] = model.config
    exported['source_checkpoint'] = str(args.source)
    exported['source_checkpoint_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        torch.save(exported, handle)
    with output_manifest.open('xb') as handle:
        handle.write(manifest)
    print(json.dumps({'checkpoint': str(args.output),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'bytes': args.output.stat().st_size,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'source_step': checkpoint['step']}, indent=2))


if __name__ == '__main__':
    main()
