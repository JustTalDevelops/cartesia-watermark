"""Build duration-valid, voice-disjoint actual-endpoint training examples.

Only observed API booleans become labels. No model-provenance labels or
unqueried edited files are substituted. Reserved test voices are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf

from remover import ROOT


def build_native_manifest():
    old = json.loads((ROOT/'results/encoder_oracle_r1.manifest.json').read_text())
    native = json.loads((ROOT/'audio/remover_validation/plan.json').read_text())['items']
    reserved_plans = [ROOT/f'results/remover_test_batch{i}.plan.json' for i in (1, 3)]
    forbidden = {r['voice'] for p in reserved_plans for r in json.loads(p.read_text())['items']}
    validation_voices = {r['voice'] for r in old['validation']} | {r['voice'] for r in native}
    source_voices = {r['path']: r['voice'] for r in native}
    for language in ('es', 'ja'):
        source_voices[f'audio/remover_validation/{language}_control.wav'] = next(
            r['voice'] for r in native if r['language'] == language)
    plans = {n: json.loads((ROOT/f'audio/corpus{n}_plan.json').read_text()) for n in (5, 6, 7)}
    for split in ('train', 'validation'):
        source_voices.update({r['path']: r['voice'] for r in old[split]})

    def voice_for(source):
        if source in source_voices:
            return source_voices[source]
        path = Path(source)
        corpus = {'corpus5': 5, 'corpus6': 6, 'corpus7': 7, 'h2h4': 7, 'sealed4': 7}.get(path.parent.name)
        if corpus is not None:
            return plans[corpus][int(path.stem.split('_')[1])][2]
        return None

    quality = {}
    for pattern in ('remover_*_validation.json', 'remover_*_h2h4.json', 'remover_*quality.json'):
        for path in (ROOT/'results').glob(pattern):
            # The earlier stopped low-pass test remains uninspected and unused.
            if 'wide6_testbatch1' in path.name:
                continue
            d = json.loads(path.read_text())
            if not isinstance(d, dict):
                continue
            if d.get('output_sha256') and isinstance(d.get('quality'), dict) and d['quality'].get('pesq_wb') is not None:
                quality[d['output_sha256']] = d['quality']['pesq_wb']
            for r in d.get('clips', []):
                if r.get('output_sha256') and r.get('pesq_wb') is not None:
                    quality[r['output_sha256']] = r['pesq_wb']

    entries = {}
    skipped = {'short_or_padded': 0, 'reserved': 0, 'unknown_voice': 0}

    def add(path, label, voice, kind, provenance, expected_hash=None, padding=0, reference=None):
        if voice is None:
            skipped['unknown_voice'] += 1
            return
        if voice in forbidden:
            skipped['reserved'] += 1
            return
        path = Path(path)
        full = ROOT/path
        info = sf.info(full)
        if padding or info.duration < 5:
            skipped['short_or_padded'] += 1
            return
        digest = hashlib.sha256(full.read_bytes()).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise ValueError(f'API-labeled file changed: {path}')
        references = []
        if reference is not None:
            references = [str((ROOT/reference).resolve().relative_to(ROOT.resolve()))]
        if digest in entries:
            if entries[digest]['label'] != label:
                raise ValueError(f'Conflicting actual labels for {path}')
            entries[digest]['provenance'].append(provenance)
            entries[digest]['reference_sources'] = sorted(set(entries[digest].get('reference_sources', [])) | set(references))
            return
        pesq = quality.get(digest)
        if pesq is not None and pesq >= 4.4:
            kind, weight = 'high_quality_edit', 8.0
        else:
            weight = 2.0 if kind == 'remover_api' else (.5 if kind == 'transform' else 1.)
        entries[digest] = {'path': str(path), 'sha256': digest, 'label': label, 'voice': voice,
            'kind': kind, 'weight': weight, 'seconds': info.duration,
            'split': 'validation' if voice in validation_voices else 'train',
            'reference_pesq': pesq, 'provenance': [provenance], 'reference_sources': references}

    for split in ('train', 'validation'):
        for r in old[split]:
            add(r['path'], r['label'], r['voice'], r['kind'],
                'results/encoder_oracle_r1.manifest.json', r['sha256'], r.get('padding', 0))
    for path in sorted((ROOT/'results').glob('remover_*cartesia.json')):
        if 'wide6_testbatch1' in path.name or 'test_batch3' in path.name:
            continue
        d = json.loads(path.read_text())
        for r in d.get('clips', []):
            source = r.get('source')
            if not source:
                continue
            voice = voice_for(source)
            if isinstance(r.get('original'), bool):
                add(source, r['original'], voice, 'natural', str(path.relative_to(ROOT)), r.get('source_sha256'), reference=source)
            if isinstance(r.get('edited'), bool) and r.get('output'):
                add(r['output'], r['edited'], voice, 'remover_api', str(path.relative_to(ROOT)), r.get('output_sha256'), reference=source)
    result = {split: [r for r in entries.values() if r['split'] == split] for split in ('train', 'validation')}
    assert not ({r['voice'] for r in result['train']} & {r['voice'] for r in result['validation']})
    result['policy'] = {'minimum_native_seconds': 5, 'padded_examples_excluded': True,
        'labels': 'Only actual observed API responses', 'reserved_voice_count': len(forbidden),
        'skipped': skipped, 'native_development_voices_held_out_of_oracle_training': True}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/api_oracle_native_r2.manifest.json')
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise ValueError('Manifest already frozen')
    data = build_native_manifest()
    out.write_text(json.dumps(data, indent=2))
    for split in ('train', 'validation'):
        rows = data[split]
        print(split, len(rows), 'voices', len({r['voice'] for r in rows}),
              'positives', sum(r['label'] for r in rows),
              'high-quality edits', sum(r['kind'] == 'high_quality_edit' for r in rows))
    print(data['policy'])
