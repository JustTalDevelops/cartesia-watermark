"""Score every eligible latent treatment and retain the entire declared panel."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest
from experiments.check_ticodec_latents import data
from experiments.ticodec_panel_report import validated as validate_receipts
from experiments.ticodec_latent_generation_report import validated_inventory

FORMATTER_SHA256 = digest(Path(__file__))


def load(path):
    return json.loads(Path(path).read_text())


def identifier(row):
    return row['id']+'__'+row['method']


def validated(declaration_path, detector_path):
    return validate_receipts(declaration_path, detector_path, builder=data)


def evaluation_plan(declaration_path, detector_path):
    declaration, _ = validated(declaration_path, detector_path)
    F = Path(declaration['completed_study_file']).parent
    original_plan = F/'evaluation.plan.json'
    study = load(declaration['completed_study_file'])
    if digest(original_plan) != study['endpoint_receipts']['baseline']['plan_sha256']:
        raise ValueError('Authored references differ from the original completed study')
    originals = {i['id']: i for i in load(original_plan)['items']}
    sources = {s['id']: s for s in declaration['sources']}
    items = []
    for row in declaration['treatments']:
        if row['detector_eligible']:
            original, source = originals[row['id']], sources[row['id']]
            items.append({'id': identifier(row), 'source': source['file'], 'sha256': source['sha256'],
                'output': row['file'], 'language': original['language'], 'text': original['text']})
    return {'declaration_sha256': digest(declaration_path), 'detector_sha256': digest(detector_path),
        'evaluation_wrapper_sha256': FORMATTER_SHA256, 'all_new_eligible_including_detected': True,
        'prior_qualified_exact_pilot_reused_without_rescoring': True, 'items': items}


def score(declaration_path, detector_path, output):
    from experiments.codec_profile_evaluation import score as frozen_score
    from experiments.probe_codec_frames import without_http
    if digest(Path(__file__)) != FORMATTER_SHA256:
        raise ValueError('Loaded evaluation wrapper changed')
    declaration, _ = validated(declaration_path, detector_path)
    plan = evaluation_plan(declaration_path, detector_path)
    plan_path, cache = output.parent/'evaluation.plan.json', output.parent/'evaluation-cache'
    if output.exists() or plan_path.exists() or cache.exists():
        raise ValueError('Never overwrite or automatically restart existing evaluation work')
    output.parent.mkdir(parents=True, exist_ok=True); cache.mkdir()
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    seeds = {}
    for original in sorted((Path(declaration['completed_study_file']).parent/'scores/cache').glob('*.json')):
        checksum, target = digest(original), cache/original.name
        shutil.copyfile(original, target)
        if digest(original) != checksum or digest(target) != checksum:
            raise ValueError('Original transcription cache changed while copying')
        seeds[original.name] = checksum
    (output.parent/'evaluation-cache-seed.json').write_text(json.dumps(seeds, indent=2)+'\n')
    with without_http(): frozen_score(plan_path, output, cache, originals_only=False)
    validated(declaration_path, detector_path)
    if digest(Path(__file__)) != FORMATTER_SHA256:
        raise ValueError('Evaluation wrapper changed during scoring')


def export(declaration_path, detector_path, evaluation_path):
    from codec_content import assess
    from codec_text import paired_error, text_units
    from experiments.codec_profile_cache import read_recognition
    from experiments.codec_profile_evaluation import evaluation_metrics, quality_criteria
    from experiments.ticodec_pilot_report import export as pilot_export
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    if digest(Path(__file__)) != FORMATTER_SHA256: raise ValueError('Loaded formatter changed')
    evaluation_sha = digest(evaluation_path)
    declaration, endpoint = validated(declaration_path, detector_path)
    F = Path(declaration['completed_study_file']).parent
    D, panel = F.parent, Path(declaration['generation_file']).parent
    plan = evaluation_plan(declaration_path, detector_path)
    plan_path = evaluation_path.parent/'evaluation.plan.json'
    evaluated, original_eval, original_voice = load(evaluation_path), load(F/'original_evaluation.json'), load(F/'original_voice.json')
    needed = {m for i in plan['items'] for m in evaluation_metrics(i['language'])}
    if (load(plan_path) != plan or evaluated.get('complete') is not True or evaluated.get('error_type') or
            evaluated.get('originals_only') is not False or evaluated.get('candidate_outputs_read') is not True or
            evaluated['plan_sha256'] != digest(plan_path) or
            evaluated['checkpoints'] != {m: original_eval['checkpoints'][m] for m in needed}):
        raise ValueError('Complete the fixed reference plan with unchanged checkpoint roles')
    expected = {(i['id'], m) for i in plan['items'] for m in evaluation_metrics(i['language'])}
    scores = {(c['id'], c['metric']): c for c in evaluated['clips']}
    if len(scores) != len(evaluated['clips']) or set(scores) != expected:
        raise ValueError('Every eligible treatment needs every declared reference measurement')
    originals = {(c['id'], c['metric']): c for c in original_eval['clips']}
    cases = {c['id']: c for c in load(declaration['completed_study_file'])['cases']}
    generated = load(declaration['generation_file'])
    raw = {(c['id'], t['mode']): t for c in generated['cases'] for t in c['treatments']}
    public_generation, _ = validated_inventory(D)
    identity, authored = generated['recognition_configuration_sha256'], {i['id']: i for i in plan['items']}
    normalizer, rows, pilot = BasicTextNormalizer(remove_diacritics=False), [], None
    for item, observed in zip(declaration['treatments'], endpoint['treatments']):
        old, treatment = cases[item['id']], raw[(item['id'], item['method'])]
        row = {k: item[k] for k in ('id', 'method', 'sha256', 'processing_status', 'endpoint_phase_status')}
        row.update(language=old['language'], source_sha256=old['source_sha256'], source_format=old['source_format'],
            speech_assessment=treatment['speech_assessment'], runtime_quality_guard=treatment.get('runtime_quality_guard'),
            qualified_spent_transition=False)
        if item['endpoint_phase_status'] == 'qualified_pilot_reused':
            pilot = pilot or pilot_export(D)
            row.update(prior_pilot_endpoint_sha256=pilot['endpoint_sha256'], prior_pilot_evaluation_sha256=pilot['evaluation_sha256'],
                qualified_spent_transition=pilot['qualified_spent_transition'], new_detector_requests_for_treatment=0)
        elif item['detector_eligible']:
            reference = authored[identifier(item)]
            before = read_recognition(F/'recognition/originals', old['source_sha256'], identity)
            after = read_recognition(panel/'recognition/processed', item['sha256'], identity)
            speech = assess(before, after)
            if speech != treatment['speech_assessment'] or not speech['content_guard_passed']:
                raise ValueError('Actual recognition differs from the processing assessment')
            guards = (speech['primary_guard'], speech['secondary_guard'])
            truth = text_units(reference['text'], normalizer, old['language'])
            errors = {name: paired_error(truth, before[name], old['language']) for name in guards}
            if errors != old['original_guard_calibration']['original_reference_error']:
                raise ValueError('Original guard reference calibration changed')
            deltas = {}
            for name in guards:
                actual = paired_error(truth, after[name], old['language'])
                deltas[name] = None if actual is None or errors[name] is None else actual-errors[name]
            eval_deltas, measurements = {}, []
            for metric in evaluation_metrics(old['language']):
                current, original = scores[(reference['id'], metric)], originals[(old['id'], metric)]
                if (current['source_sha256'] != old['source_sha256'] or current['output_sha256'] != item['sha256'] or
                        current['original'] != original['original'] or current['status'] != 'scored'):
                    raise ValueError('Reference measurement refers to another whole file or calibration')
                errors[metric] = original['original']['reference_error']
                actual = current['processed']['reference_error']
                delta = None if actual is None or errors[metric] is None else actual-errors[metric]
                if delta != current['reference_error_delta']: raise ValueError('Reference error delta changed')
                eval_deltas[metric] = delta; measurements.append(current)
            quality = treatment['runtime_quality_guard']
            receipt_path = panel/'runtime_voice'/old['id']/(item['sha256']+'.json')
            receipt = load(receipt_path); clip = receipt['clips'][0]
            original = next(c for c in original_voice['clips'] if c['id'] == old['id'])
            if (digest(receipt_path) != quality['receipt_sha256'] or receipt.get('complete') is not True or
                    receipt['metric'] != original_voice['metric'] or clip['original_dnsmos'] != original['original_dnsmos'] or
                    clip['source_sha256'] != old['source_sha256'] or clip['output_sha256'] != item['sha256'] or
                    clip['speaker_minimum'] != quality['speaker_minimum'] or clip['dnsmos_delta']['overall'] != quality['dnsmos_overall_delta']):
                raise ValueError('Voice observation differs from the operational measurement')
            qualification = quality_criteria(old['language'], speech['paired_disagreement'], deltas, eval_deltas, errors,
                clip['speaker_minimum'], clip['dnsmos_delta']['overall'], True,
                old['original_guard_calibration']['language_matches_request'])
            row.update(qualification, detected=observed['detected'], original_reference_error=errors,
                guard_reference_delta=deltas, independent_evaluation=measurements, voice_estimates=clip,
                identical_file_observations_reused=observed['identical_file_observations_reused'],
                qualified_spent_transition=not observed['detected'] and qualification['quality_qualified'])
        rows.append(row)
    validated(declaration_path, detector_path)
    if digest(evaluation_path) != evaluation_sha or digest(Path(__file__)) != FORMATTER_SHA256:
        raise ValueError('Evaluation or formatter changed during curation')
    controls = [{k: c[k] for k in ('id', 'control_status', 'control_sha256')} for c in public_generation['cases']]
    return {'kind': 'Spent-source TiCodec latent development results', 'formatter_source_sha256': FORMATTER_SHA256,
        'declaration_sha256': digest(declaration_path), 'endpoint_sha256': digest(detector_path), 'evaluation_sha256': evaluation_sha,
        'generation_sha256': declaration['generation_sha256'], 'completed_study_sha256': declaration['completed_study_sha256'],
        'code_files_sha256': declaration['code_files_sha256'], 'evaluation_checkpoints': evaluated['checkpoints'],
        'model': public_generation['model'], 'planned_spent_sources': declaration['planned_sources'],
        'planned_generation_treatments': len(rows), 'new_endpoint_treatments': declaration['eligible_treatments'],
        'exact_quantized_controls': controls, 'qualified_spent_sources': len({r['id'] for r in rows if r['qualified_spent_transition']}),
        'qualified_treatments': sum(r['qualified_spent_transition'] for r in rows), 'treatments': rows,
        'http_attempts': [{k: v for k, v in c.items() if k != 'file'} for c in endpoint['http_attempts']],
        'checks': [{k: v for k, v in c.items() if k != 'file'} for c in endpoint['checks']], 'prior_qualified_pilot': pilot,
        'new_tts_requests': 0, 'fresh_efficacy_sources': 0, 'completed_19_of_44_result_changed': False,
        'human_listening_approval': False, 'physical_erasure_proven': False,
        'limits': ['These are spent development recordings; repeated methods and controls are not independent sources.',
                   'Voice estimates guide eligibility; Whisper checkpoints share ancestry.',
                   'Negative detector replies and estimated quality cannot establish physical erasure or universal reliability.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'export'))
    parser.add_argument('--declaration', type=Path, required=True); parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'score': score(args.declaration, args.detector, args.evaluation)
    else:
        if args.output is None or args.output.exists(): parser.error('Choose a new --output')
        result = export(args.declaration, args.detector, args.evaluation)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('TICODEC_LATENT_QUALIFIED', result['qualified_spent_sources'], '/', result['planned_spent_sources'])
