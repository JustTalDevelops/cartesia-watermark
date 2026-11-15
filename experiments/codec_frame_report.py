"""Evaluate all eligible spent frame treatments after the complete detector panel.

The score command reuses the completed study's frozen evaluator. Curated exports
contain numeric evidence and retain every treatment, including processing failures.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from experiments.check_codec_frames import declaration_data, digest

FORMATTER_SHA256 = digest(__file__)


def load(path):
    return json.loads(Path(path).read_text())


def treatment_id(row):
    return row['id']+'__offset'+str(row['offset_native_samples'])


def validated_detector(declaration_path, detector_path):
    declaration, measured = load(declaration_path), load(detector_path)
    if ({k:v for k,v in declaration.items() if k != 'declared_at'} !=
            declaration_data(declaration['generation_file'], declaration['completed_study_file'])):
        raise ValueError('Declared inventory changed')
    if (measured.get('complete') is not True or measured.get('error_type') or
            measured.get('declaration_sha256') != digest(declaration_path)):
        raise ValueError('Complete the declared detector panel before reference evaluation')
    expected = []
    def pair(row, role):
        for ordinal in (1,2):
            expected.append({'id':row['id'], 'role':role, 'ordinal':ordinal,
                'sha256':row['sha256'], 'file':row['file'],
                **({'offset_native_samples':row['offset_native_samples']} if role=='candidate' else {})})
    pair(declaration['sources'][0], 'opening_positive_control')
    seen, treatments = {}, []
    for row in declaration['treatments']:
        result = dict(row)
        if row['detector_eligible']:
            checksum = row['sha256']
            reused = checksum in seen
            if not reused:
                pair(row, 'candidate')
                observation = measured['observations'][checksum]
                if (type(observation.get('detected')) is not bool or
                        observation['id'] != row['id'] or observation['offset_native_samples'] != row['offset_native_samples']):
                    raise ValueError('Candidate observations refer to different treatments')
                seen[checksum] = observation
            result.update(detector_observation_group=checksum, detected=seen[checksum]['detected'],
                          identical_file_observations_reused=reused)
        treatments.append(result)
    pair(declaration['sources'][-1], 'closing_positive_control')
    if (measured['treatments'] != treatments or measured['observations'] != seen or
            len(expected) != len(measured['checks']) or len(expected) != len(measured['http_attempts']) or
            len(expected) != declaration['maximum_http_attempts']):
        raise ValueError('Every declared treatment and actual request must remain exactly once')
    starts = []
    for planned, check, attempt in zip(expected, measured['checks'], measured['http_attempts']):
        if (any(check[k] != value or attempt[k] != value for k,value in planned.items()) or
                check['http_status'] != 200 or attempt['http_status'] != 200 or
                type(check.get('detected')) is not bool or attempt.get('error_type')):
            raise ValueError('Incomplete or mismatched exact-file detector receipts')
        detected = check['detected']
        historical = declaration['historical_exact_file_observations'].get(check['sha256'])
        if ((check['role'] != 'candidate' and detected is not True) or
                (historical is not None and historical != detected) or
                (check['role'] == 'candidate' and detected != seen[check['sha256']]['detected'])):
            raise ValueError('Contradictory detector receipt or failed positive control')
        starts.append(datetime.fromisoformat(attempt['started_at']))
    if (starts[0] < datetime.fromisoformat(declaration['declared_at']) or
            any((b-a).total_seconds() < 79.9 for a,b in zip(starts,starts[1:]))):
        raise ValueError('Detector declaration timing or shared pacing changed')
    return declaration, measured


def evaluation_plan(declaration_path, detector_path):
    declaration, measured = validated_detector(declaration_path, detector_path)
    study_path = Path(declaration['completed_study_file'])
    old_plan = study_path.parent/'evaluation.plan.json'
    study = load(study_path)
    if digest(old_plan) != study['endpoint_receipts']['baseline']['plan_sha256']:
        raise ValueError('Authored references differ from the completed study')
    authored = {r['id']:r for r in load(old_plan)['items']}
    items = []
    for row in declaration['treatments']:
        if not row['detector_eligible']:
            continue
        original = authored[row['id']]
        items.append({'id':treatment_id(row), 'source':str((ROOT/original['source']).resolve()),
            'sha256':original['sha256'], 'language':original['language'], 'text':original['text'], 'output':row['file']})
    return {'declaration_sha256':digest(declaration_path), 'detector_sha256':digest(detector_path),
            'reference_evaluation_after_detector_complete':True, 'all_eligible_including_detected':True, 'items':items}


def score(declaration_path, detector_path, output):
    from experiments.codec_profile_evaluation import score as frozen_score
    from experiments.probe_codec_frames import without_http
    if digest(__file__) != FORMATTER_SHA256:
        raise ValueError('Loaded evaluation wrapper changed')
    declaration, _ = validated_detector(declaration_path, detector_path)
    plan = evaluation_plan(declaration_path, detector_path)
    plan_path = output.parent/'evaluation.plan.json'
    cache = output.parent/'evaluation-cache'
    if output.exists() or plan_path.exists() or cache.exists():
        raise ValueError('Existing evaluation work is never silently overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    cache.mkdir()
    seeds = {}
    original_cache = Path(declaration['completed_study_file']).parent/'scores/cache'
    for original in sorted(original_cache.glob('*.json')):
        checksum = digest(original)
        target = cache/original.name
        shutil.copyfile(original, target)
        if digest(original) != checksum or digest(target) != checksum:
            raise ValueError('Completed-study transcription cache changed while copying')
        seeds[original.name] = checksum
    (output.parent/'evaluation-cache-seed.json').write_text(json.dumps(seeds, indent=2)+'\n')
    with without_http():
        frozen_score(plan_path, output, cache, originals_only=False)
    validated_detector(declaration_path, detector_path)
    if digest(__file__) != FORMATTER_SHA256:
        raise ValueError('Evaluation wrapper changed during scoring')


def export(declaration_path, detector_path, evaluation_path):
    from codec_content import assess
    from codec_text import paired_error, text_units
    from experiments.codec_profile_cache import read_recognition
    from experiments.codec_profile_evaluation import evaluation_metrics, quality_criteria
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    if digest(__file__) != FORMATTER_SHA256:
        raise ValueError('Loaded formatter changed')
    declaration, endpoint = validated_detector(declaration_path, detector_path)
    expected_plan = evaluation_plan(declaration_path, detector_path)
    plan_path = evaluation_path.parent/'evaluation.plan.json'
    evaluation = load(evaluation_path)
    study_dir = Path(declaration['completed_study_file']).parent
    generation_dir = Path(declaration['generation_file']).parent
    study = load(declaration['completed_study_file'])
    generation = load(declaration['generation_file'])
    originals, original_voice = load(study_dir/'original_evaluation.json'), load(study_dir/'original_voice.json')
    if (load(plan_path) != expected_plan or evaluation.get('complete') is not True or
            evaluation['plan_sha256'] != digest(plan_path) or evaluation['checkpoints'] != originals['checkpoints']):
        raise ValueError('Use the complete fixed all-eligible evaluation and original checkpoints')
    inputs = {r['id']:r for r in expected_plan['items']}
    expected_pairs = {(r['id'],m) for r in expected_plan['items'] for m in evaluation_metrics(r['language'])}
    measured = {(r['id'],r['metric']):r for r in evaluation['clips']}
    if len(measured) != len(evaluation['clips']) or set(measured) != expected_pairs:
        raise ValueError('Every eligible treatment needs every declared evaluator')
    original_eval = {(r['id'],r['metric']):r for r in originals['clips']}
    old_cases = {r['id']:r for r in study['cases']}
    generated = {(c['id'],t['offset_native_samples']):t for c in generation['cases'] for t in c['treatments']}
    identity = generation['recognition_configuration_sha256']
    normalizer = BasicTextNormalizer(remove_diacritics=False)
    results = []
    for declared, observed in zip(declaration['treatments'],endpoint['treatments']):
        case = old_cases[declared['id']]
        treatment = generated[(declared['id'],declared['offset_native_samples'])]
        model = treatment['model']
        row = {k:declared[k] for k in ('id','offset_native_samples','processing_status','sha256')}
        row.update(language=case['language'], source_sha256=case['source_sha256'],
            historical_baseline_comparison=treatment.get('historical_baseline_comparison'),
            content_guard_passed=model['content_guard_passed'], paired_disagreement=model['paired_disagreement'],
            runtime_quality_guard=treatment.get('runtime_quality_guard'), qualified_spent_transition=False)
        if declared['detector_eligible']:
            item = inputs[treatment_id(declared)]
            before = read_recognition(study_dir/'recognition/originals',case['source_sha256'],identity)
            after = read_recognition(generation_dir/'recognition/processed',declared['sha256'],identity)
            assessment = assess(before,after)
            if any(model.get(k) != v for k,v in assessment.items()) or not assessment['content_guard_passed']:
                raise ValueError('Processing assessment differs from its actual recognition cache')
            guards = (assessment['primary_guard'],assessment['secondary_guard'])
            truth = text_units(item['text'],normalizer,case['language'])
            errors, deltas = {}, {}
            for name in guards:
                errors[name] = paired_error(truth,before[name],case['language'])
                actual = paired_error(truth,after[name],case['language'])
                deltas[name] = None if actual is None or errors[name] is None else actual-errors[name]
            if errors != case['original_guard_calibration']['original_reference_error']:
                raise ValueError('Original guard reference calibration changed')
            eval_deltas, eval_rows = {}, []
            for metric in evaluation_metrics(case['language']):
                result, original = measured[(item['id'],metric)], original_eval[(case['id'],metric)]
                if (result['source_sha256'] != case['source_sha256'] or result['output_sha256'] != declared['sha256'] or
                        result['original'] != original['original'] or result['status'] != 'scored'):
                    raise ValueError('Evaluation refers to another whole file or original calibration')
                errors[metric] = original['original']['reference_error']
                actual = result['processed']['reference_error']
                delta = None if actual is None or errors[metric] is None else actual-errors[metric]
                if result['reference_error_delta'] != delta:
                    raise ValueError('Evaluation delta changed')
                eval_deltas[metric] = delta;eval_rows.append(result)
            quality = treatment['runtime_quality_guard']
            receipt = generation_dir/'runtime_voice'/case['id']/(declared['sha256']+'.json')
            voice = load(receipt)
            original = next(r for r in original_voice['clips'] if r['id']==case['id'])
            clip = voice['clips'][0]
            if (digest(receipt) != quality['receipt_sha256'] or voice.get('complete') is not True or
                    voice['metric'] != original_voice['metric'] or clip['original_dnsmos'] != original['original_dnsmos'] or
                    clip['source_sha256'] != case['source_sha256'] or clip['output_sha256'] != declared['sha256'] or
                    clip['speaker_minimum'] != quality['speaker_minimum'] or
                    clip['dnsmos_delta']['overall'] != quality['dnsmos_overall_delta']):
                raise ValueError('Voice evidence differs from its operational gate')
            qualification = quality_criteria(case['language'],assessment['paired_disagreement'],deltas,eval_deltas,errors,
                clip['speaker_minimum'],clip['dnsmos_delta']['overall'],True,
                case['original_guard_calibration']['language_matches_request'])
            row.update(qualification, detected=observed['detected'], original_reference_error=errors,
                guard_reference_delta=deltas, independent_evaluation=eval_rows, voice_estimates=clip,
                identical_file_observations_reused=observed['identical_file_observations_reused'],
                qualified_spent_transition=not observed['detected'] and qualification['quality_qualified'])
        results.append(row)
    successful = {r['id'] for r in results if r['qualified_spent_transition']}
    return {'kind':'Spent-source codec frame-alignment development results',
        'formatter_source_sha256':FORMATTER_SHA256,
        'generation_sha256':declaration['generation_sha256'], 'declaration_sha256':digest(declaration_path),
        'detector_sha256':digest(detector_path), 'evaluation_sha256':digest(evaluation_path),
        'completed_study_sha256':declaration['completed_study_sha256'], 'code_files_sha256':declaration['code_files_sha256'],
        'planned_spent_sources':declaration['planned_sources'], 'planned_treatments':len(results),
        'eligible_treatments':declaration['eligible_treatments'], 'unique_eligible_files':declaration['unique_eligible_files'],
        'qualified_spent_sources':len(successful), 'qualified_treatments':sum(r['qualified_spent_transition'] for r in results),
        'treatments':results,
        'http_attempts':[{k:v for k,v in r.items() if k!='file'} for r in endpoint['http_attempts']],
        'checks':[{k:v for k,v in r.items() if k!='file'} for r in endpoint['checks']],
        'fresh_efficacy_sources':0, 'new_tts_requests':0, 'completed_study_result_changed':False,
        'human_listening_approval':False, 'physical_erasure_proven':False,
        'limits':['All recordings were spent development cases; offset treatments and repeats are not independent sources.',
                  'Operational voice estimates guide eligibility; Whisper checkpoints share ancestry.',
                  'Negative responses and estimated quality do not prove physical erasure or universal removal.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('score','export'))
    parser.add_argument('--declaration',type=Path,required=True)
    parser.add_argument('--detector',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.phase == 'score':
        score(args.declaration,args.detector,args.evaluation)
    else:
        if args.output is None or args.output.exists():
            parser.error('Export requires a new --output')
        value = export(args.declaration,args.detector,args.evaluation)
        args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        print('FRAME_RESULTS',value['qualified_spent_sources'],'/',value['planned_spent_sources'])
