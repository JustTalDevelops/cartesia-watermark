"""Combine two trained editors, selecting additional edits by voice fidelity.

The second editor is the quality baseline. A blend is used only when its
encoded audio meets per-recording PESQ/ESTOI limits relative to that baseline.
No watermark detector is consulted during inference.
"""
import numpy as np
import torch
from torch import nn


class QualityBlend(nn.Module):
    def __init__(self, component_configs, fractions=(.25, .125), max_pesq_drop=.03,
                 max_estoi_drop=.001, pesq_floor=4.50, min_snr_db=30.):
        super().__init__()
        from remover import Remover
        if len(component_configs) != 2:
            raise ValueError('Quality blend requires two standard editors')
        if any(c.get('post_lowpass_hz') is not None for c in component_configs):
            raise ValueError('Quality blend does not support broad output filtering')
        if not fractions or any(not 0 < f <= 1 for f in fractions):
            raise ValueError('Blend fractions must be in (0, 1]')
        if min(max_pesq_drop, max_estoi_drop) < 0 or min_snr_db < 0:
            raise ValueError('Quality budgets must be nonnegative')
        self.components = nn.ModuleList([Remover(**c) for c in component_configs])
        self.config = dict(component_configs=component_configs, fractions=sorted(set(fractions), reverse=True),
                           max_pesq_drop=max_pesq_drop, max_estoi_drop=max_estoi_drop,
                           pesq_floor=pesq_floor, min_snr_db=min_snr_db)
        self.last_quality_decision = None

    @staticmethod
    def _metrics(reference, audio, sample_rate):
        from pesq import pesq, PesqError
        from pystoi import stoi
        from remover import resample
        ref = reference[:, None] if reference.ndim == 1 else reference
        out = audio[:, None] if audio.ndim == 1 else audio
        # Longer recordings are checked throughout, with overlapping windows.
        size = min(len(ref), 20*sample_rate)
        starts = sorted(set([*range(0, max(1, len(ref)-size+1), max(1, size//2)), len(ref)-size]))
        scores = []
        for channel in range(ref.shape[1]):
            for start in starts:
                a, b = ref[start:start+size, channel], out[start:start+size, channel]
                if np.mean(a*a) < 1e-10:
                    continue
                p = float(pesq(16000, resample(a, sample_rate, 16000), resample(b, sample_rate, 16000),
                               'wb', on_error=PesqError.RETURN_VALUES))
                if p <= 0:
                    raise ValueError('Unscorable speech window')
                e = float(stoi(a, b, sample_rate, extended=True))
                if not np.isfinite(e):
                    raise ValueError('Unscorable intelligibility window')
                scores.append((p, e))
        if not scores:
            raise ValueError('No scoreable speech')
        return np.asarray(scores)

    def process_native(self, audio, sample_rate, strength, chunk_seconds):
        from remover import process_audio
        # Match FLOAT WAV component encoding used by the verified prototypes.
        first, baseline = [process_audio(m, audio, sample_rate, strength, chunk_seconds)
                           .astype(np.float32).astype(np.float64) for m in self.components]
        self.last_quality_decision = {'fraction': 0., 'reason': 'quality_baseline'}
        if len(audio) < sample_rate:
            return baseline
        try:
            base_scores = self._metrics(audio, baseline, sample_rate)
        except (ValueError, RuntimeError):
            self.last_quality_decision['reason'] = 'unscorable_quality_baseline'
            return baseline
        for fraction in self.config['fractions']:
            candidate = baseline + fraction*(first-baseline)
            axis = 0 if candidate.ndim == 2 else None
            delta = candidate-audio
            budget = np.sqrt(np.mean(audio*audio, axis=axis))*10**(-self.config['min_snr_db']/20)*min(strength, 1.)
            gain = np.minimum(1., budget/np.maximum(np.sqrt(np.mean(delta*delta, axis=axis)), 1e-20))
            candidate = (audio+delta*gain).astype(np.float32).astype(np.float64)
            if np.max(np.abs(candidate)) >= 1.:
                continue
            try:
                scores = self._metrics(audio, candidate, sample_rate)
            except (ValueError, RuntimeError):
                continue
            minimum_pesq = np.maximum(base_scores[:, 0]-self.config['max_pesq_drop'],
                                      np.minimum(base_scores[:, 0], self.config['pesq_floor']))
            if (np.all(scores[:, 0] >= minimum_pesq) and
                    np.all(scores[:, 1] >= base_scores[:, 1]-self.config['max_estoi_drop'])):
                self.last_quality_decision = {'fraction': fraction, 'reason': 'quality_limits_met',
                    'worst_pesq_delta': float(np.min(scores[:, 0]-base_scores[:, 0])),
                    'worst_estoi_delta': float(np.min(scores[:, 1]-base_scores[:, 1]))}
                return candidate
        return baseline


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from remover import load_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-pesq-drop', type=float, default=.03)
    parser.add_argument('--selection-voice', action='append', default=[])
    args = parser.parse_args()
    output_manifest = args.output.with_suffix('.manifest.json')
    if args.output.exists() or output_manifest.exists():
        parser.error('Output exists; choose a new version')
    models, manifests, provenance = [], [], []
    calibration, inherited = set(args.selection_voice), set()
    for path in (args.first, args.baseline):
        model, checkpoint = load_model(path)
        if checkpoint.get('format') != 'cartesia-remover-v1':
            parser.error('Components must be standard trained editors')
        manifest_path = path.with_suffix('.manifest.json')
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checkpoint['manifest_sha256']:
            parser.error('Component manifest hash mismatch')
        data = json.loads(raw)
        inherited |= {r['voice'] for r in data['train']} | set(data.get('initialization_train_voices', []))
        calibration |= (set(data.get('oracle_calibration_voices', [])) |
                        set(checkpoint.get('teacher_config', {}).get('training_voices', [])) |
                        set(checkpoint.get('teacher_config', {}).get('calibration_voices', [])))
        models.append(model)
        manifests.append(data)
        provenance.append({'checkpoint': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'manifest_sha256': hashlib.sha256(raw).hexdigest()})
    for split in ('train', 'validation'):
        if {r['voice'] for r in manifests[0][split]} != {r['voice'] for r in manifests[1][split]}:
            parser.error('Source voice splits differ; audit their lineage first')
    data = manifests[1]
    data.update(initialization_train_voices=sorted(inherited), oracle_calibration_voices=sorted(calibration),
                component_provenance=provenance, quality_selection_voices=sorted(set(args.selection_voice)))
    if {r['voice'] for r in data['validation']} & (inherited | calibration):
        parser.error('Validation overlaps component training or selection voices')
    blended = QualityBlend([m.config for m in models], max_pesq_drop=args.max_pesq_drop)
    for component, source in zip(blended.components, models):
        component.load_state_dict(source.state_dict())
    encoded = json.dumps(data, indent=2).encode()
    checkpoint = {'format': 'cartesia-quality-blend-v1', 'model_config': blended.config,
                  'state_dict': blended.state_dict(), 'step': 0,
                  'manifest_sha256': hashlib.sha256(encoded).hexdigest(),
                  'component_provenance': provenance,
                  'limitations': 'Quality-gated combination of two trained editors; no guarantee of watermark removal'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        torch.save(checkpoint, handle)
    with output_manifest.open('xb') as handle:
        handle.write(encoded)
    print(json.dumps({'checkpoint': str(args.output),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'parameters': sum(p.numel() for p in blended.parameters()),
                      'policy': blended.config}), flush=True)


if __name__ == '__main__':
    main()
