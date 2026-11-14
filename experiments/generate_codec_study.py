"""Generate a sealed codec-study plan using an existing, locked budget ledger.

Never initializes or resets the ledger, and never repeats an attempted request.
Raw provider audio and whole canonical input receipts are kept separately.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, directory):
        self.path, self.lock_path = directory/'budget.json', directory/'budget.lock'
        if not self.path.is_file():
            raise ValueError('An existing authorized budget ledger is required; it is never initialized automatically')

    def mutate(self, action):
        with self.lock_path.open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            budget = json.loads(self.path.read_text())
            if (budget['authorized_cap_usd'] != 50 or budget['reservation_rate_usd_per_million_credits'] != 65 or
                    sum(r['reserved_credits'] for r in budget['requests']) != budget['reserved_credits']):
                raise ValueError('Budget authorization, rate or existing reservations changed; inspect the ledger')
            result = action(budget)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(budget, indent=2)+'\n')
            temporary.replace(self.path)
            return result

    def reserve(self, item):
        def action(budget):
            if any(r['id'] == item['id'] for r in budget['requests']):
                raise ValueError('A prior reservation exists; inspect its receipt rather than retrying')
            credits = 2*len(item['text'].encode('utf8'))
            maximum = min(budget['maximum_reserved_credits'], int(50*1_000_000/65))
            if budget['reserved_credits']+credits > maximum:
                raise ValueError('The shared $50 limit would be exceeded; no request sent')
            row = {'id': item['id'], 'model': item['model'], 'voice': item['voice'],
                   'characters': len(item['text']), 'utf8_bytes': len(item['text'].encode('utf8')),
                   'reserved_credits': credits, 'status': 'reserved', 'attempted': False, 'reserved_at': now()}
            budget['reserved_credits'] += credits
            budget['requests'].append(row)
            return row
        return self.mutate(action)

    def update(self, key, **values):
        if {'id', 'model', 'voice', 'characters', 'utf8_bytes', 'reserved_credits', 'reserved_at'} & values.keys():
            raise ValueError('Reservation identity and cost fields are immutable')
        def action(budget):
            row = next(r for r in budget['requests'] if r['id'] == key)
            row.update(values)
        self.mutate(action)


def canonicalize(raw, output, item):
    audio, rate = sf.read(raw, dtype='float32')
    if rate != item['sample_rate'] or audio.ndim != 1 or not np.isfinite(audio).all() or not 5 <= len(audio)/rate <= 300:
        raise ValueError('Provider returned an unexpected whole-recording format or duration')
    if item.get('expected_duration_seconds'):
        low, high = item['expected_duration_seconds']
        if not low <= len(audio)/rate <= high:
            raise ValueError('Provider recording did not meet its predeclared long-duration slot')
    gains = item.get('stereo_derivation_gains')
    if gains is not None:
        audio = np.stack([audio*gain for gain in gains], axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        sf.write(handle, audio, rate, format='FLAC', subtype='PCM_24')
    decoded, actual_rate = sf.read(output, dtype='float32')
    if actual_rate != rate or decoded.shape != audio.shape or np.max(np.abs(decoded-audio)) > 2**-23:
        raise ValueError('Canonical encoding changed the complete grid or exceeded PCM24 quantization')
    return {'sample_rate': rate, 'samples': len(audio), 'seconds': len(audio)/rate,
            'channels': 1 if audio.ndim == 1 else audio.shape[1]}


def generate(plan_path, freeze_path, directory):
    plan, freeze = json.loads(plan_path.read_text()), json.loads(freeze_path.read_text())
    if freeze['generation_plan_sha256'] != digest(plan_path) or freeze['fresh_generation_started'] is not False:
        raise ValueError('Generation plan does not match the sealed prospective freeze')

    def check_sources():
        if not all(digest(ROOT/name) == checksum for name, checksum in freeze['source_files_sha256'].items()):
            raise ValueError('Frozen source or evaluation code changed; no request sent')

    check_sources()
    ledger = Ledger(directory)
    path = directory/'codec_fresh/generation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(path.read_text()) if path.exists() else {
        'generation_plan_sha256': digest(plan_path), 'freeze_sha256': digest(freeze_path), 'requests': [], 'complete': False}
    if receipt['generation_plan_sha256'] != digest(plan_path) or receipt['freeze_sha256'] != digest(freeze_path):
        raise ValueError('Existing receipt belongs to another freeze')
    sys.path.insert(0, str(ROOT/'tools'))
    import cartesia
    for item in plan['items']:
        check_sources()
        raw, canonical = ROOT/item['raw_output'], ROOT/item['canonical_output']
        previous = next((r for r in receipt['requests'] if r['id'] == item['id']), None)
        if previous:
            if digest(raw) != previous['raw_sha256'] or digest(canonical) != previous['sha256']:
                raise ValueError('Previously generated audio changed')
            continue
        if raw.exists() or canonical.exists():
            raise ValueError('Unreceipted audio exists; inspect it rather than charging for a repeat')
        ledger.reserve(item)
        ledger.update(item['id'], attempted=True, status='attempted', attempted_at=now())
        try:
            raw.parent.mkdir(parents=True, exist_ok=True)
            headers = cartesia.tts(item['text'], raw, model=item['model'], voice=item['voice'],
                                   language=item['language'], sample_rate=item['sample_rate'],
                                   container='wav', encoding='pcm_f32le')
            info = canonicalize(raw, canonical, item)
            row = {'id': item['id'], 'source': item['canonical_output'], 'sha256': digest(canonical),
                   'raw_source': item['raw_output'], 'raw_sha256': digest(raw), 'created_at': now(),
                   'voice': item['voice'], 'model': item['model'], 'language': item['language'],
                   'marked_provenance': item['marked_provenance'], **info,
                   'request_id': headers.get('x-request-id', headers.get('request-id'))}
            ledger.update(item['id'], status='completed', output_sha256=row['raw_sha256'], canonical_output_sha256=row['sha256'])
            receipt['requests'].append(row)
            path.write_text(json.dumps(receipt, indent=2)+'\n')
            print('GENERATED', item['id'], round(info['seconds'], 2), 'seconds', flush=True)
        except Exception as error:
            match = re.search(r'tts (\d{3})', str(error))
            ledger.update(item['id'], status='failed_or_uncertain', error_type=type(error).__name__,
                          http_status=int(match.group(1)) if match else None)
            print('GENERATION_STOPPED', item['id'], type(error).__name__, flush=True)
            return 2
    receipt['complete'] = True
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    print('GENERATION_COMPLETE', len(receipt['requests']), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    args = parser.parse_args()
    return generate(args.plan, args.freeze, args.directory)


if __name__ == '__main__':
    raise SystemExit(main())
