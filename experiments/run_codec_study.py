"""Run frozen whole-file checks without reference text entering reconstruction.

Baseline and candidate phases have separate receipts. Repeat responses refer to
the same exact file, never additional independent observations. API failures or
inconsistent repeated responses stop the phase for inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'tools'))
os.environ['DETECT_PROXY'] = ''
os.environ['DETECT_INTERVAL'] = '80'
import cartesia
from reconstruction_worker import ReconstructionWorker
from verify_reconstruction import NotCleared, reconstruct_verified


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(plan_path, freeze_path, directory, phase):
    plan, freeze = load(plan_path), load(freeze_path)
    if digest(plan_path) != freeze['generation_plan_sha256']:
        raise ValueError('Plan differs from its prospective freeze')

    def check_sources():
        if not all(digest(ROOT/name) == checksum for name, checksum in freeze['source_files_sha256'].items()):
            raise ValueError('Frozen source/evaluation code changed')

    check_sources()
    generation_path = directory/'generation.json'
    generation = load(generation_path)
    if (not generation['complete'] or generation['generation_plan_sha256'] != digest(plan_path) or
            generation['freeze_sha256'] != digest(freeze_path)):
        raise ValueError('A complete generation receipt matching this freeze is required')
    receipts = {i['id']: i for i in generation['requests']}
    if set(receipts) != {i['id'] for i in plan['items']}:
        raise ValueError('Every planned request must remain in the study')
    materialized_path = directory/'evaluation.plan.json'
    if not materialized_path.exists():
        materialized = {'generation_plan_sha256': digest(plan_path), 'generation_receipt_sha256': digest(generation_path),
                        'freeze_sha256': digest(freeze_path), 'items': []}
        for item in plan['items']:
            receipt = receipts[item['id']]
            materialized['items'].append(dict(item, source=receipt['source'], sha256=receipt['sha256'], output=item['verified_output']))
        with materialized_path.open('x') as handle:
            json.dump(materialized, handle, indent=2, ensure_ascii=False)
    materialized = load(materialized_path)
    if materialized['generation_receipt_sha256'] != digest(generation_path):
        raise ValueError('Generation receipt changed after materialization')
    if phase == 'materialize':
        print(materialized_path, flush=True)
        return 0
    baseline_path = directory/'baseline.json'
    if phase == 'candidates':
        baseline = load(baseline_path)
        if not baseline['complete'] or baseline['freeze_sha256'] != digest(freeze_path):
            raise ValueError('Complete original detector baseline is required before candidate checks')
        for metric in ('small', 'turbo', 'mms'):
            report = load(directory/'scores'/('originals_'+metric+'.json'))
            if (not report['complete'] or not report['originals_only'] or
                    report['candidate_outputs_read'] or report['plan_sha256'] != digest(materialized_path) or
                    len(report['clips']) != len(plan['items'])):
                raise ValueError('All original-only recognizer calibrations are required before selection')
    report_path = directory/(phase+'.json')
    if report_path.exists():
        raise ValueError('Inspect an existing phase receipt instead of silently repeating observations')
    report = {'phase': phase, 'freeze_sha256': digest(freeze_path), 'generation_receipt_sha256': digest(generation_path),
              'http_attempts': [], 'checks': [], 'cases': [], 'complete': False, 'started_at': now()}
    save(report_path, report)
    current = {}
    original_post = cartesia.requests.post

    def counted(*args, **kwargs):
        if len(report['http_attempts']) >= freeze['maximum_http_attempts_per_phase'][phase]:
            raise ValueError('Frozen phase request ceiling reached')
        entry = dict(current, started_at=now())
        report['http_attempts'].append(entry)
        save(report_path, report)
        try:
            response = original_post(*args, **kwargs)
        except Exception as error:
            entry['exception'] = type(error).__name__
            save(report_path, report)
            raise
        entry['http_status'] = response.status_code
        save(report_path, report)
        return response

    cartesia.requests.post = counted

    def check(path, key, role, ordinal):
        check_sources()
        checksum = digest(path)
        current.clear()
        current.update(id=key, role=role, ordinal=ordinal, sha256=checksum)
        status, response = cartesia.detect(path)
        if digest(path) != checksum:
            raise ValueError('Detector input changed')
        entry = dict(current, http_status=status, checked_at=now())
        if isinstance(response, dict) and type(response.get('detected')) is bool:
            entry['detected'] = response['detected']
        report['checks'].append(entry)
        save(report_path, report)
        if status != 200 or 'detected' not in entry:
            raise RuntimeError('Detector unavailable; phase stopped without retry')
        print(phase, key, role, ordinal, entry['detected'], flush=True)
        return status, response

    def repeated(path, key, role, expected=None):
        first = check(path, key, role, 1)[1]['detected']
        second = check(path, key, role, 2)[1]['detected']
        if first != second or (expected is not None and first != expected):
            raise RuntimeError('Repeated detector response is inconsistent with the phase control')
        return first

    worker = None
    try:
        if phase == 'baseline':
            for item in plan['items']:
                receipt = receipts[item['id']]
                source = ROOT/receipt['source']
                if digest(source) != receipt['sha256']:
                    raise ValueError('Canonical source changed')
                detected = repeated(source, item['id'], 'original')
                report['cases'].append({'id': item['id'], 'source_sha256': receipt['sha256'],
                                        'marked_provenance': item['marked_provenance'], 'detected': detected})
                save(report_path, report)
        else:
            positives = [i for i in plan['items'] if i['marked_provenance'] and
                         next(c['detected'] for c in baseline['cases'] if c['id'] == i['id'])]
            if not positives:
                raise ValueError('No measured-positive marked controls; candidate phase cannot establish transitions')
            opening, closing = positives[0], positives[-1]
            repeated(ROOT/receipts[opening['id']]['source'], opening['id'], 'opening_positive_control', True)
            worker = ReconstructionWorker(ROOT/'.cache/resemble-venv/bin/python', ROOT/'tools/reconstruct_codec.py')
            for item in plan['items']:
                receipt = receipts[item['id']]
                source, output, audit = ROOT/receipt['source'], ROOT/item['verified_output'], ROOT/item['audit_output']
                if digest(source) != receipt['sha256']:
                    raise ValueError('Canonical source changed')

                def detector(path):
                    # Use actual audit ordinals; original controls never enter this count.
                    record = load(audit)
                    attempt = record['attempts'][-1]
                    ordinal = attempt['checks'][-1]['ordinal']
                    role = 'processed_'+attempt['preset']
                    status, response = check(path, item['id'], role, ordinal)
                    if response['detected']:
                        # Positive counterexamples receive a same-file confirmation too.
                        second = check(path, item['id'], role+'_positive_confirmation', 2)[1]['detected']
                        if second is not True:
                            raise RuntimeError('Inconsistent candidate responses; no result selected')
                    return status, response

                try:
                    audit_record = reconstruct_verified(source, output, audit, tuple(plan['candidate_presets']),
                                                        worker=worker, detector=detector, confirmations=2)
                except NotCleared:
                    audit_record = load(audit)
                case = {k: audit_record[k] for k in ('source_sha256', 'status', 'attempts', 'quality_verified')}
                case.update(id=item['id'], marked_provenance=item['marked_provenance'],
                            original_detected=next(c['detected'] for c in baseline['cases'] if c['id'] == item['id']))
                for key in ('output_sha256', 'selected_preset', 'negative_checks'):
                    if key in audit_record:
                        case[key] = audit_record[key]
                report['cases'].append(case)
                save(report_path, report)
                print('CASE_COMPLETE', item['id'], case['status'], flush=True)
            repeated(ROOT/receipts[closing['id']]['source'], closing['id'], 'closing_positive_control', True)
        check_sources()
        report.update(complete=True, completed_at=now())
        save(report_path, report)
        return 0
    except Exception as error:
        report.update(stopped_at=now(), error_type=type(error).__name__)
        save(report_path, report)
        raise
    finally:
        cartesia.requests.post = original_post
        if worker is not None:
            worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability/codec_fresh')
    parser.add_argument('--phase', choices=('materialize', 'baseline', 'candidates'), required=True)
    args = parser.parse_args()
    return run(args.plan, args.freeze, args.directory, args.phase)


if __name__ == '__main__':
    raise SystemExit(main())
