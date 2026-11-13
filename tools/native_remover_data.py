"""Prepare native-length remover training without padding short utterances.

Retain original clips >=5 s and join existing same-voice/model-family renders
with the same 0.3 s pause used by the native evaluation corpus. No TTS/API calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from remover import ROOT, resample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/remover_oracle_long.manifest.json')
    parser.add_argument('--teacher', type=Path, default=ROOT/'results/raw_oracle_native_r2.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/remover_native_generator.manifest.json')
    args = parser.parse_args()
    source, teacher, output = args.source, args.teacher, args.output
    if output.exists():
        raise ValueError('Native training split already frozen')
    original = json.loads(source.read_text())
    teacher_voices = set(json.loads(teacher.read_text())['training_voices'])
    data = {key: value for key, value in original.items() if key not in ('train', 'validation')}
    data.update(purpose='Native>=5soriginalandjoinedtraining;nozeropadding;raworacletrainingvoicesexcludedfromgeneratorvalidation',
                source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                teacher_config_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest())
    folder = ROOT/'audio/remover_native_generator'
    folder.mkdir(exist_ok=True)
    for split in ('train', 'validation'):
        selected, seen = [], set()
        for row in original[split]:
            if split == 'validation' and row['voice'] in teacher_voices:
                continue
            path = ROOT/row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Original training audio changed')
            info = sf.info(path)
            if info.duration >= 5 and row['sha256'] not in seen:
                selected.append(dict(row, native_seconds=info.duration))
                seen.add(row['sha256'])
            prefix = path.stem.rsplit('_', 1)[0]
            models = ('sonic-3.6', 'sonic-preview') if row['marked'] else ('sonic-3.5', 'sonic-3')
            pair = [path.with_name(prefix+'_'+model+'.wav') for model in models]
            if not all(p.exists() for p in pair):
                continue
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in pair]
            identity = hashlib.sha256(('nativejoin.3:'+':'.join(hashes)).encode()).hexdigest()
            if identity in seen:
                continue
            a, rate = sf.read(pair[0])
            b, other_rate = sf.read(pair[1])
            if a.ndim != 1 or b.ndim != 1:
                raise ValueError('Expected mono training renders')
            b = resample(b, other_rate, rate)
            joined = np.concatenate((a, np.zeros(round(rate*.3)), b)).astype(np.float32)
            if len(joined)/rate < 5:
                continue
            target = folder/(split+'_'+identity[:20]+'.wav')
            if not target.exists():
                sf.write(target, joined, rate, subtype='FLOAT')
            else:
                existing, existing_rate = sf.read(target)
                if existing_rate != rate or not np.array_equal(existing, joined.astype(np.float64)):
                    raise ValueError('Existing joined training file changed')
            selected.append(dict(row, path=str(target.relative_to(ROOT)),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(), model='+'.join(models),
                native_seconds=len(joined)/rate, added_pause_seconds=.3,
                source_paths=[str(p.relative_to(ROOT)) for p in pair], source_sha256=hashes))
            seen.add(identity)
        data[split] = selected
    train = {r['voice'] for r in data['train']}
    validation = {r['voice'] for r in data['validation']}
    assert not train & validation
    assert not validation & teacher_voices
    reserved = json.loads((ROOT/'results/remover_test_batch3.plan.json').read_text())
    assert not (train | validation) & {r['voice'] for r in reserved['items']}
    output.write_text(json.dumps(data, indent=2))
    for split in ('train', 'validation'):
        rows = data[split]
        print(split, len(rows), 'voices', len({r['voice'] for r in rows}),
              'minseconds', min(r['native_seconds'] for r in rows),
              'marked', sum(r['marked'] for r in rows))


if __name__ == '__main__':
    main()
