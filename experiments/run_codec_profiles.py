"""Run a prospectively frozen language-profile study with original-only caches.

Reference text is used by the separate original/quality scoring phases. Candidate
selection uses detected language, duration, processing guards and actual detector
responses. A supplied in-process model adapter retains only the active source
between fallbacks, then clears it; the warmed models last for this phase.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
os.environ['DETECT_PROXY'] = ''
os.environ['DETECT_INTERVAL'] = '80'
import cartesia
from codec_audio import digest
from codec_content import ContentRecognizer
from codec_profiles import method_order, PROFILE_VERSION
from codec_text import paired_error, text_units
from experiments.codec_profile_cache import configuration, store_recognition, read_recognition, StudyRecognizer
from experiments.codec_profile_evaluation import evaluation_metrics
from experiments.score_codec_study import save
from reconstruct_codec import CodecRuntime
from verify_reconstruction import reconstruct_verified, NotCleared


def load(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen_sources(freeze):
    if not all(digest(ROOT/name) == checksum for name, checksum in freeze['source_files_sha256'].items()):
        raise ValueError('Frozen source/evaluation code changed')


def materialized(plan_path, freeze_path, directory):
    from experiments.run_codec_study import run
    return run(plan_path, freeze_path, directory, 'materialize')


def validate_materialized(plan, freeze, freeze_path, directory):
    generation = load(directory/'generation.json')
    if (plan['freeze_sha256'] != digest(freeze_path) or
            plan['generation_plan_sha256'] != freeze['generation_plan_sha256'] or
            plan['generation_receipt_sha256'] != digest(directory/'generation.json') or
            not generation['complete'] or generation['freeze_sha256'] != digest(freeze_path) or
            generation['generation_plan_sha256'] != freeze['generation_plan_sha256']):
        raise ValueError('Materialized inputs do not bind a completed frozen generation receipt')
    requested = {r['id']: r for r in generation['requests']}
    if len(requested) != len(generation['requests']) or len(plan['items']) != len(requested) or set(requested) != {i['id'] for i in plan['items']}:
        raise ValueError('Every planned generated source must remain exactly once')
    if any(requested[i['id']]['sha256'] != i['sha256'] or requested[i['id']]['source'] != i['source'] for i in plan['items']):
        raise ValueError('Materialized sources differ from their generation receipts')


def original_guards(plan_path, freeze_path, directory):
    freeze, plan = load(freeze_path), load(plan_path)
    frozen_sources(freeze)
    validate_materialized(plan, freeze, freeze_path, directory)
    path = directory/'original_guards.json'
    if path.exists():
        raise ValueError('Inspect the existing original-only calibration instead of repeating it')
    rec = ContentRecognizer()
    identity = configuration(rec.metadata)
    report = {'freeze_sha256': digest(freeze_path), 'plan_sha256': digest(plan_path),
              'originals_only': True, 'candidate_outputs_read': False, 'configuration_sha256': identity,
              'recognizers': rec.metadata, 'clips': [], 'complete': False}
    save(path, report)
    for item in plan['items']:
        frozen_sources(freeze)
        source = ROOT/item['source']
        if digest(source) != item['sha256'] or digest(plan_path) != report['plan_sha256']:
            raise ValueError('Original plan or complete source changed')
        before = rec.transcribe(source)
        if digest(source) != item['sha256']:
            raise ValueError('Original changed during recognition')
        cached = store_recognition(directory/'recognition/originals', item['sha256'], identity, before)
        # Recognition is already fixed before the authored reference is accessed.
        language = before['language']
        matching = language == item['language'] and before['supported']
        names = (before.get('primary', 'turbo'), before.get('secondary', 'mms'))
        truth = text_units(item['text'], rec.normalizer, item['language'])
        errors = {n: paired_error(truth, before.get(n, []), item['language']) if matching else None for n in names}
        report['clips'].append({'id': item['id'], 'source_sha256': item['sha256'], 'detected_language': language,
                                'language_matches_request': matching, 'primary_guard': names[0], 'secondary_guard': names[1],
                                'original_reference_error': errors, 'recognition_cache_sha256': digest(cached)})
        save(path, report)
        print('ORIGINAL_PROFILE_GUARDS', item['id'], language, matching, flush=True)
    report.update(complete=True, completed_at=now())
    save(path, report)


class RuntimeAdapter:
    def __init__(self, runtime):
        self.runtime = runtime

    def run(self, source, output, preset, metadata, timeout):
        return self.runtime.process_file(source, output, preset, metadata, retain_job=True)


def endpoint_phase(plan_path, freeze_path, directory, phase):
    freeze, plan = load(freeze_path), load(plan_path)
    frozen_sources(freeze)
    validate_materialized(plan, freeze, freeze_path, directory)
    path = directory/(phase+'.json')
    if path.exists():
        raise ValueError('Existing service observations are never silently repeated')
    plan_sha, freeze_sha = digest(plan_path), digest(freeze_path)
    baseline = None
    runtime = None
    receipts = {}
    if phase == 'candidates':
        for name in ('baseline.json', 'original_guards.json', 'original_evaluation.json', 'original_voice.json'):
            value = load(directory/name)
            if not value['complete'] or (name != 'baseline.json' and
                    (not value['originals_only'] or value['candidate_outputs_read'])):
                raise ValueError('Complete baseline and original-only calibrations are required before selection')
            if value['plan_sha256'] != plan_sha:
                raise ValueError('A required preflight belongs to a different input plan')
            receipts[name] = digest(directory/name)
        baseline, originals = load(directory/'baseline.json'), load(directory/'original_guards.json')
        expected_ids = {i['id'] for i in plan['items']}
        if (len(baseline['cases']) != len(expected_ids) or {c['id'] for c in baseline['cases']} != expected_ids or
                baseline['freeze_sha256'] != digest(freeze_path)):
            raise ValueError('The baseline must retain every source under this freeze')
        evaluation = load(directory/'original_evaluation.json')
        expected_pairs = {(i['id'], metric) for i in plan['items'] for metric in evaluation_metrics(i['language'])}
        if (len(evaluation['clips']) != len(expected_pairs) or
                {(c['id'], c['metric']) for c in evaluation['clips']} != expected_pairs):
            raise ValueError('Every independent original checkpoint must be calibrated')
        voice = load(directory/'original_voice.json')
        if len(voice['clips']) != len(expected_ids) or {c['id'] for c in voice['clips']} != expected_ids:
            raise ValueError('Every original needs its voice estimate before processing')
        source_digests = {c['source_sha256']: c['recognition_cache_sha256'] for c in originals['clips']}
        if len(originals['clips']) != len(expected_ids) or {c['id'] for c in originals['clips']} != expected_ids:
            raise ValueError('Every planned original must remain in preflight')
        rec = StudyRecognizer(directory/'recognition/originals', directory/'recognition/processed',
                              source_digests=source_digests)
        if rec.identity != originals['configuration_sha256']:
            raise ValueError('Processing recognition differs from original calibration')
        runtime = CodecRuntime(recognizer=rec)
    report = {'phase': phase, 'freeze_sha256': freeze_sha, 'plan_sha256': plan_sha,
              'profile_version': PROFILE_VERSION, 'required_preflight_sha256': receipts,
              'http_attempts': [], 'checks': [], 'cases': [], 'complete': False, 'started_at': now(),
              'reference_text_used_for_selection': False, 'evaluation_metrics_used_for_selection': False}
    save(path, report)
    current = {}
    original_post = cartesia.requests.post

    def counted(*args, **kwargs):
        if not current or len(report['http_attempts']) >= freeze['maximum_http_attempts_per_phase'][phase]:
            raise ValueError('Unexpected request or frozen phase request ceiling reached')
        attempt = dict(current, started_at=now())
        report['http_attempts'].append(attempt)
        save(path, report)
        try:
            response = original_post(*args, **kwargs)
        except Exception as error:
            attempt['exception'] = type(error).__name__
            save(path, report)
            raise
        attempt['http_status'] = response.status_code
        save(path, report)
        return response

    cartesia.requests.post = counted

    def check(audio, case, role, ordinal):
        frozen_sources(freeze)
        if digest(plan_path) != plan_sha or digest(freeze_path) != freeze_sha or any(
                digest(directory/n) != sha for n, sha in receipts.items()):
            raise ValueError('Plan, freeze or completed source preflight changed')
        checksum = digest(audio)
        current.clear(); current.update(id=case, role=role, ordinal=ordinal, sha256=checksum)
        status, response = cartesia.detect(audio)
        if digest(audio) != checksum:
            raise ValueError('Whole-file detector input changed')
        row = dict(current, http_status=status, checked_at=now())
        if isinstance(response, dict) and type(response.get('detected')) is bool:
            row['detected'] = response['detected']
        report['checks'].append(row); save(path, report)
        if status != 200 or 'detected' not in row:
            raise RuntimeError('Detector unavailable; phase stopped without retry')
        print('PROFILE_ENDPOINT', phase, case, role, ordinal, row['detected'], flush=True)
        return status, response

    def repeated(audio, case, role, expected=None):
        first = check(audio, case, role, 1)[1]['detected']
        second = check(audio, case, role, 2)[1]['detected']
        if first != second or (expected is not None and first != expected):
            raise RuntimeError('Repeated detector response conflicts with its control')
        return first

    try:
        if phase == 'baseline':
            for item in plan['items']:
                source = ROOT/item['source']
                if digest(source) != item['sha256']:
                    raise ValueError('Canonical original changed')
                detected = repeated(source, item['id'], 'original')
                report['cases'].append({'id': item['id'], 'source_sha256': item['sha256'],
                                        'marked_provenance': item['marked_provenance'], 'detected': detected})
                save(path, report)
        else:
            positive = [i for i in plan['items'] if i['marked_provenance'] and
                        next(c['detected'] for c in baseline['cases'] if c['id'] == i['id'])]
            if not positive:
                raise ValueError('No measured-positive marked source controls')
            repeated(ROOT/positive[0]['source'], positive[0]['id'], 'opening_positive_control', True)
            for item in plan['items']:
                source, output, audit = ROOT/item['source'], ROOT/item['output'], ROOT/item['audit_output']
                if digest(source) != item['sha256']:
                    raise ValueError('Canonical source changed')
                before = runtime.recognizer.transcribe(source)
                info = sf.info(source)
                orders = method_order(before['language'], info.frames/info.samplerate) if before['supported'] else ('codec-native',)

                def detector(candidate):
                    attempt = load(audit)['attempts'][-1]
                    ordinal = attempt['checks'][-1]['ordinal']
                    role = 'processed_'+attempt['preset']
                    status, response = check(candidate, item['id'], role, ordinal)
                    if response['detected']:
                        if check(candidate, item['id'], role+'_positive_confirmation', 2)[1]['detected'] is not True:
                            raise RuntimeError('Candidate positive confirmation conflicts')
                    return status, response

                try:
                    selected = reconstruct_verified(source, output, audit, orders, worker=RuntimeAdapter(runtime),
                                                    detector=detector, confirmations=2, skip_duplicate_candidates=True)
                except NotCleared:
                    selected = load(audit)
                finally:
                    runtime.clear_job()
                row = {k: selected[k] for k in ('source_sha256', 'status', 'attempts', 'quality_verified')}
                row.update(id=item['id'], detected_language=before['language'], method_order=list(orders),
                           marked_provenance=item['marked_provenance'],
                           original_detected=next(c['detected'] for c in baseline['cases'] if c['id'] == item['id']))
                for key in ('output_sha256', 'selected_preset', 'negative_checks'):
                    if key in selected:
                        row[key] = selected[key]
                report['cases'].append(row); save(path, report)
                print('PROFILE_CASE_COMPLETE', item['id'], row['status'], flush=True)
            repeated(ROOT/positive[-1]['source'], positive[-1]['id'], 'closing_positive_control', True)
        frozen_sources(freeze)
        report.update(complete=True, completed_at=now()); save(path, report)
    except Exception as error:
        report.update(error_type=type(error).__name__, stopped_at=now()); save(path, report)
        raise
    finally:
        cartesia.requests.post = original_post
        if runtime is not None:
            runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--phase', choices=('materialize', 'originals', 'baseline', 'candidates'), required=True)
    args = parser.parse_args()
    if args.phase == 'materialize':
        materialized(args.plan, args.freeze, args.directory)
    elif args.phase == 'originals':
        original_guards(args.plan, args.freeze, args.directory)
    else:
        endpoint_phase(args.plan, args.freeze, args.directory, args.phase)
