"""Declare all completed latent cells after the preceding detector phases finish."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest
from experiments import check_codec_frames as endpoint
from experiments.ticodec_latent_generation_report import validated_inventory
from experiments.ticodec_pilot_report import export as pilot_export


def load(path):
    return json.loads(Path(path).read_text())


def previous_receipts(directory, study):
    """Freeze complete earlier endpoint phases; never snapshot a growing report."""
    known = endpoint.previous_observations(study)
    files = {}
    for panel in ('codec_frame_development', 'ticodec_joint'):
        declaration = directory/panel/'endpoint.declaration.json'
        result = directory/panel/'detector.json'
        declared, measured = load(declaration), load(result)
        if (measured.get('complete') is not True or measured.get('error_type') or
                measured['declaration_sha256'] != digest(declaration) or
                len(measured['checks']) != declared['maximum_http_attempts'] or
                len(measured['http_attempts']) != declared['maximum_http_attempts']):
            raise ValueError('Finish both preceding endpoint panels before declaring the latent phase')
        expected = []
        def pair(row, role):
            for ordinal in (1, 2):
                expected.append({'id': row['id'], 'sha256': row['sha256'], 'file': row['file'],
                    'role': role, 'ordinal': ordinal,
                    **{k: row[k] for k in ('method', 'offset_native_samples') if k in row}})
        pair(declared['sources'][0], 'opening_positive_control')
        seen = set()
        for row in declared['treatments']:
            if row['detector_eligible'] and row['sha256'] not in seen:
                pair(row, 'candidate')
                seen.add(row['sha256'])
        pair(declared['sources'][-1], 'closing_positive_control')
        if len(expected) != len(measured['checks']):
            raise ValueError('Earlier endpoint requests differ from their complete declared inventory')
        starts = []
        for planned, check, attempt in zip(expected, measured['checks'], measured['http_attempts']):
            if (type(check.get('detected')) is not bool or check['http_status'] != 200 or
                    attempt['http_status'] != 200 or attempt.get('error_type') or
                    any(check[k] != v or attempt[k] != v for k, v in planned.items()) or
                    (check['role'] != 'candidate' and check['detected'] is not True)):
                raise ValueError('Earlier endpoint receipts are incomplete or contradictory')
            checksum, detected = check['sha256'], check['detected']
            if checksum in known and known[checksum] != detected:
                raise ValueError('Historical exact-file replies conflict')
            known[checksum] = detected
            starts.append(datetime.fromisoformat(attempt['started_at']))
        if (starts[0] < datetime.fromisoformat(declared['declared_at']) or
                any((b-a).total_seconds() < 79.9 for a, b in zip(starts, starts[1:]))):
            raise ValueError('Earlier endpoint declaration timing or shared pacing changed')
        files[str(declaration.resolve())] = digest(declaration)
        files[str(result.resolve())] = digest(result)
    return known, files


def data(generation_path, study_path):
    generation_path, study_path = Path(generation_path).resolve(), Path(study_path).resolve()
    directory = study_path.parent.parent
    if (generation_path != directory/'ticodec_latent_development/generation.json' or
            study_path != directory/'codec_profiles_fresh/curated_complete_report.json'):
        raise ValueError('Use the canonical completed latent panel and its original study')
    study = load(study_path)
    known, historical_files = previous_receipts(directory, study)
    public, inventory = validated_inventory(directory)
    pilot = pilot_export(directory)
    if pilot['qualified_spent_transition'] is not True:
        raise ValueError('Retain the original independently qualified pilot')
    for check in load(directory/'ticodec_pilot/result.json')['checks']:
        checksum, detected = check['sha256'], check['detected']
        if checksum in known and known[checksum] != detected:
            raise ValueError('Historical pilot reply conflicts')
        known[checksum] = detected
    assessment = dict(inventory['assessment_files_sha256'])
    assessment.update(historical_files)
    for name in ('ticodec_pilot.declaration.json', 'ticodec_pilot/result.json',
                 'ticodec_pilot/quality/evaluation.json', 'ticodec_pilot/quality/evaluation.plan.json'):
        path = directory/name
        assessment[str(path)] = digest(path)
    sources, treatments = inventory['sources'], inventory['treatments']
    for row in treatments:
        source = next(s for s in sources if s['id'] == row['id'])
        if (not 5 <= source['format']['samples']/source['format']['sample_rate'] <= 300 or
                max(Path(row['file']).stat().st_size, Path(source['file']).stat().st_size) > 32_000_000):
            raise ValueError('Keep whole files within the service upload limits')
        if (row['detector_eligible'] and row['sha256'] == pilot['candidate_sha256'] and
                row['id'] == pilot['case_id'] and source['sha256'] == pilot['source_sha256']):
            row.update(detector_eligible=False, endpoint_phase_status='qualified_pilot_reused',
                       prior_pilot_endpoint_sha256=pilot['endpoint_sha256'], prior_quality_qualified=True,
                       prior_candidate_detected=False, new_requests_for_this_treatment=0)
    code = dict(public['code_files_sha256'])
    for name in ('experiments/check_codec_frames.py', 'experiments/check_ticodec_latents.py',
                 'experiments/ticodec_latent_generation_report.py', 'experiments/ticodec_generation_report.py',
                 'experiments/ticodec_pilot_report.py',
                 'experiments/ticodec_latent_report.py', 'experiments/ticodec_panel_report.py',
                 'docs/codec-ticodec-latent-evaluation.md'):
        code[name] = digest(ROOT/name)
    pending = [t for t in treatments if t['detector_eligible']]
    unique = len({t['sha256'] for t in pending})
    return {'kind': 'Declared spent-source TiCodec latent endpoint panel',
        'observation_report_kind': 'Spent-source TiCodec latent detector observations',
        'progress_prefix': 'TICODEC_LATENT_DETECTOR', 'generation_file': str(generation_path),
        'generation_sha256': digest(generation_path), 'completed_study_file': str(study_path),
        'completed_study_sha256': digest(study_path), 'code_files_sha256': code,
        'assessment_files_sha256': assessment, 'sources': sources, 'treatments': treatments,
        'historical_exact_file_observations': known, 'planned_sources': len(sources),
        'planned_generation_treatments': len(treatments), 'paired_quantized_controls': len(sources),
        'operationally_eligible_treatments': sum(t['processing_status'] == 'processing_eligible' for t in treatments),
        'qualified_pilot_treatments_reused': sum(t['endpoint_phase_status'] == 'qualified_pilot_reused' for t in treatments),
        'eligible_treatments': len(pending), 'unique_eligible_files': unique, 'maximum_http_attempts': 2*unique+4,
        'minimum_detector_start_interval_seconds': 80, 'repetitions_per_exact_file': 2,
        'opening_source_id': sources[0]['id'], 'closing_source_id': sources[-1]['id'],
        'new_tts_requests': 0, 'fresh_efficacy_sources': 0,
        'reference_evaluation_order': 'After complete endpoint receipts; score every newly eligible treatment including positives',
        'human_listening_approval': False, 'physical_erasure_proven': False}


def declare(generation_path, study_path, path):
    if path.exists(): raise ValueError('Never overwrite a declared endpoint inventory')
    value = data(generation_path, study_path)
    value['declared_at'] = endpoint.now()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle: handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return value


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
        result = declare(args.generation, args.completed_study, args.declaration)
        print('TICODEC_LATENTS_DECLARED', result['planned_generation_treatments'], 'rows;',
              result['unique_eligible_files'], 'new files;', result['maximum_http_attempts'], 'HTTP maximum')
    else:
        endpoint.run(args.declaration, args.output, builder=data)
