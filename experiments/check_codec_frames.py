"""Declare and check the complete spent-source frame panel, without synthesis.

Declaration requires completed generation. Detection checks every eligible exact
file twice, deduplicates identical bytes and stops on errors or contradictions.
Reference evaluation is a subsequent phase and cannot change this inventory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'tools'))
os.environ['DETECT_PROXY'] = ''
os.environ['DETECT_INTERVAL'] = '80'
import cartesia


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


IMPLEMENTATION_SHA256 = digest(__file__)
OFFSETS = (0, 80, 160, 240)
TERMINAL = {'processing_eligible', 'content_rejected', 'runtime_quality_rejected'}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def previous_observations(study):
    seen = {}
    for phase in study['endpoint_receipts'].values():
        for check in phase['checks']:
            checksum, detected = check['sha256'], check['detected']
            if type(detected) is not bool or (checksum in seen and seen[checksum] != detected):
                raise ValueError('Historical exact-file detector observations conflict')
            seen[checksum] = detected
    return seen


def declaration_data(generation_path, study_path):
    generation_path, study_path = Path(generation_path).resolve(), Path(study_path).resolve()
    generation = json.loads(generation_path.read_text())
    study = json.loads(study_path.read_text())
    if (generation.get('complete') is not True or generation.get('error_type') or
            generation.get('kind') != 'Spent-source codec frame-alignment generation panel' or
            generation.get('completed_study_sha256') != digest(study_path) or
            study.get('kind') != 'Frozen prospective language-profile study' or
            generation.get('new_http_requests') != 0 or generation.get('new_tts_requests') != 0):
        raise ValueError('Use the completed generation panel and its unchanged completed study')
    registry = dict(generation['code_files_sha256'])
    registry['experiments/check_codec_frames.py'] = IMPLEMENTATION_SHA256
    registry['docs/codec-frame-evaluation.md'] = digest(ROOT/'docs/codec-frame-evaluation.md')
    if any(digest(ROOT/name) != checksum for name, checksum in registry.items()):
        raise ValueError('Loaded checker or recorded generation code changed')
    failed = [c for c in study['cases'] if c['marked_provenance'] and not c['qualified_marked_transition']]
    cases = generation['cases']
    if (not failed or [c['id'] for c in cases] != [c['id'] for c in failed] or
            generation['planned_sources'] != len(failed) or
            generation['planned_treatments'] != len(failed)*len(OFFSETS) or
            generation['offsets_native_samples'] != list(OFFSETS)):
        raise ValueError('Keep every failed marked source and all four fixed offsets')
    treatments, sources, assessment_files = [], [], {}
    for name in ('evaluation.plan.json', 'original_guards.json', 'original_evaluation.json', 'original_voice.json'):
        path = study_path.parent/name
        assessment_files[str(path)] = digest(path)
    for case, old in zip(cases, failed):
        source = (ROOT/case['source']).resolve()
        source_sha = digest(source)
        info = sf.info(source)
        grid = {'samples': info.frames, 'sample_rate': info.samplerate, 'channels': info.channels}
        if (source_sha != case['source_sha256'] or source_sha != old['source_sha256'] or
                grid != old['source_format'] or old['original_detected'] is not True or
                case['original_detected'] is not True or not 5 <= info.duration <= 300 or
                source.stat().st_size > 32_000_000):
            raise ValueError('Use the unchanged complete marked sources')
        if [t['offset_native_samples'] for t in case['treatments']] != list(OFFSETS):
            raise ValueError('Keep the fixed offset order for every source')
        sources.append({'id':case['id'], 'file':str(source), 'sha256':source_sha, 'format':grid})
        original_cache = study_path.parent/'recognition/originals'/(source_sha+'.json')
        if digest(original_cache) != old['original_guard_calibration']['recognition_cache_sha256']:
            raise ValueError('Original-only recognition calibration changed')
        assessment_files[str(original_cache)] = digest(original_cache)
        for treatment in case['treatments']:
            status, model = treatment['status'], treatment['model']
            offset = treatment['offset_native_samples']
            if status not in TERMINAL or model['source_sha256'] != source_sha:
                raise ValueError('Incomplete or mismatched treatment remains in the panel')
            if offset == 0:
                expected = case['historical_native_sha256']
                comparison = 'unpaired_baseline' if expected is None else 'exact_match'
                if (treatment['historical_baseline_comparison'] != comparison or
                        (expected is not None and model['candidate_flac_sha256'] != expected)):
                    raise ValueError('Historical zero-offset parity must hold')
            row = {'id':case['id'], 'offset_native_samples':offset, 'processing_status':status,
                   'sha256':model['candidate_flac_sha256'], 'detector_eligible':status == 'processing_eligible'}
            processed_cache = generation_path.parent/'recognition/processed'/(row['sha256']+'.json')
            assessment_files[str(processed_cache)] = digest(processed_cache)
            if status != 'content_rejected':
                output = Path(treatment['output_file']).resolve()
                actual = sf.info(output)
                quality = treatment['runtime_quality_guard']
                if (not output.is_relative_to(generation_path.parent) or digest(output) != row['sha256'] or
                        treatment['output_sha256'] != row['sha256'] or model['content_guard_passed'] is not True or
                        (actual.frames, actual.samplerate, actual.channels) != (info.frames, info.samplerate, info.channels) or
                        output.stat().st_size > 32_000_000 or quality['source_sha256'] != source_sha or
                        quality['candidate_sha256'] != row['sha256'] or type(quality.get('passed')) is not bool or
                        quality['passed'] != (status == 'processing_eligible')):
                    raise ValueError('Eligible or voice-rejected output lacks its unchanged whole-file evidence')
                row['file'] = str(output)
                voice_receipt = generation_path.parent/'runtime_voice'/case['id']/(row['sha256']+'.json')
                if digest(voice_receipt) != quality['receipt_sha256']:
                    raise ValueError('Operational voice measurement changed')
                assessment_files[str(voice_receipt)] = digest(voice_receipt)
            elif model['content_guard_passed'] is not False:
                raise ValueError('A speech rejection must retain its failed processing decision')
            treatments.append(row)
    eligible = [t for t in treatments if t['detector_eligible']]
    unique = len({t['sha256'] for t in eligible})
    return {'kind':'Declared spent-source frame detector panel',
            'generation_file':str(generation_path), 'generation_sha256':digest(generation_path),
            'completed_study_file':str(study_path), 'completed_study_sha256':digest(study_path),
            'code_files_sha256':registry, 'assessment_files_sha256':assessment_files,
            'sources':sources, 'treatments':treatments,
            'historical_exact_file_observations':previous_observations(study),
            'minimum_detector_start_interval_seconds':80, 'repetitions_per_exact_file':2,
            'opening_source_id':sources[0]['id'], 'closing_source_id':sources[-1]['id'],
            'planned_sources':len(sources), 'planned_treatments':len(treatments),
            'eligible_treatments':len(eligible), 'unique_eligible_files':unique,
            'maximum_http_attempts':2*unique+4, 'new_tts_requests':0, 'fresh_efficacy_sources':0,
            'reference_evaluation_order':'After all declared detector observations',
            'human_listening_approval':False, 'physical_erasure_proven':False}


def declare(generation_path, study_path, path):
    if path.exists():
        raise ValueError('Declarations are never overwritten')
    value = declaration_data(generation_path, study_path)
    value['declared_at'] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return value


def run(declaration_path, output_path, builder=None):
    build = builder or declaration_data
    declaration = json.loads(declaration_path.read_text())
    expected = build(declaration['generation_file'], declaration['completed_study_file'])
    if {k:v for k,v in declaration.items() if k != 'declared_at'} != expected:
        raise ValueError('Declaration inventory or recorded files changed')
    if output_path.exists() or output_path.resolve() == declaration_path.resolve():
        raise ValueError('Existing service observations are never automatically repeated')
    declaration_sha = digest(declaration_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'kind':declaration.get('observation_report_kind','Spent-source frame detector observations'), 'declaration_sha256':declaration_sha,
              'complete':False, 'checks':[], 'http_attempts':[], 'observations':{},
              'treatments':declaration['treatments'], 'fresh_efficacy_sources':0, 'new_tts_requests':0,
              'independent_quality_evaluated':False, 'started_at':now()}
    save(output_path, report)
    current = {}
    original_post = cartesia.requests.post
    cartesia.MIN_INTERVAL = max(80., cartesia.MIN_INTERVAL)

    def unchanged(target, checksum):
        if (digest(declaration_path) != declaration_sha or digest(target) != checksum or
                digest(declaration['generation_file']) != declaration['generation_sha256'] or
                digest(declaration['completed_study_file']) != declaration['completed_study_sha256'] or
                any(digest(ROOT/name) != value for name,value in declaration['code_files_sha256'].items()) or
                any(digest(name) != value for name,value in declaration.get('assessment_files_sha256', {}).items())):
            raise ValueError('Declared code, inventory or exact detector input changed')

    def counted(*args, **kwargs):
        if not current or len(report['http_attempts']) >= declaration['maximum_http_attempts']:
            raise RuntimeError('Unexpected request or declared request ceiling reached')
        target, checksum = Path(current['file']), current['sha256']
        unchanged(target, checksum)
        attempt = dict(current, started_at=now())
        report['http_attempts'].append(attempt)
        save(output_path, report)
        try:
            response = original_post(*args, **kwargs)
        except Exception as error:
            attempt['error_type'] = type(error).__name__
            save(output_path, report)
            raise RuntimeError('Detector connection failed; no automatic retry') from error
        attempt['http_status'] = response.status_code
        save(output_path, report)
        return response

    def pair(row, role):
        target, checksum = Path(row['file']), row['sha256']
        replies = []
        for ordinal in (1, 2):
            unchanged(target, checksum)
            current.clear()
            current.update(id=row['id'], role=role, ordinal=ordinal, sha256=checksum, file=str(target))
            if 'offset_native_samples' in row:
                current['offset_native_samples'] = row['offset_native_samples']
            if 'method' in row:
                current['method'] = row['method']
            status, response = cartesia.detect(target)
            if status != 200 or not isinstance(response, dict) or type(response.get('detected')) is not bool:
                raise RuntimeError('Detector unavailable; no repeated request')
            unchanged(target, checksum)
            detected = response['detected']
            report['checks'].append(dict(current, detected=detected, http_status=status, checked_at=now()))
            save(output_path, report)
            historical = declaration['historical_exact_file_observations'].get(checksum)
            if ((replies and replies[0] != detected) or
                    (historical is not None and historical != detected) or
                    (role != 'candidate' and detected is not True)):
                raise RuntimeError('Contradictory exact-file observation or failed positive control')
            replies.append(detected)
            print(declaration.get('progress_prefix','FRAME_DETECTOR'), row['id'], role,
                  row.get('offset_native_samples',row.get('method')), ordinal, detected, flush=True)
        return replies[0]

    cartesia.requests.post = counted
    try:
        pair(declaration['sources'][0], 'opening_positive_control')
        for row in report['treatments']:
            if not row['detector_eligible']:
                continue
            checksum = row['sha256']
            reused = checksum in report['observations']
            if not reused:
                detected = pair(row, 'candidate')
                report['observations'][checksum] = {'detected':detected, 'id':row['id'],
                    **{key:row[key] for key in ('offset_native_samples','method') if key in row}}
            row.update(detector_observation_group=checksum,
                       detected=report['observations'][checksum]['detected'], identical_file_observations_reused=reused)
            save(output_path, report)
        pair(declaration['sources'][-1], 'closing_positive_control')
        if build(declaration['generation_file'], declaration['completed_study_file']) != expected:
            raise ValueError('Declared complete inventory changed before phase completion')
        report.update(complete=True, completed_at=now())
        save(output_path, report)
    except BaseException as error:
        report.update(error_type=type(error).__name__, stopped_at=now())
        save(output_path, report)
        raise
    finally:
        cartesia.requests.post = original_post
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    before = sub.add_parser('declare')
    before.add_argument('--generation', type=Path, required=True)
    before.add_argument('--completed-study', type=Path, required=True)
    before.add_argument('--declaration', type=Path, required=True)
    after = sub.add_parser('detect')
    after.add_argument('--declaration', type=Path, required=True)
    after.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'declare':
        value = declare(args.generation, args.completed_study, args.declaration)
        print('FRAME_DETECTOR_DECLARED', value['eligible_treatments'], 'treatments;', value['unique_eligible_files'], 'exact files')
    else:
        run(args.declaration, args.output)
