"""Declare every completed X-Codec2 and cached-repair cell before detection."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
from experiments.xcodec2_generation_report import digest, load, read_recognition, export as xcodec_export
from experiments.check_ticodec_latents import previous_receipts
from experiments import check_codec_frames as endpoint

RENDERED = {'processing_eligible', 'content_rejected', 'runtime_quality_rejected'}
UNRENDERED = {'not_needed_speech_guard_passed', 'no_compatible_mms_alignment', 'proposal_rejected'}


def validate_repair_inventory(measured, generated, declared, expected_ids):
    if (measured.get('complete') is not True or generated.get('complete') is not True or
            measured.get('error_type') or generated.get('error_type') or
            measured['planned_sources'] != 25 or measured['planned_slots'] != 50 or
            declared['planned_sources'] != 25 or declared['planned_slots'] != 50 or
            any([c['id'] for c in value['cases']] != expected_ids for value in (measured, generated, declared))):
        raise ValueError('Keep the complete 25-source, 50-slot repair inventory')
    if (declared['maximum_coverage'] != .20 or declared['maximum_regions'] != 8 or
            declared['padding_seconds'] != .12 or declared['timing_drift_limit_seconds'] != .25 or
            declared.get('reference_text_used_for_generation') is not False or
            declared.get('detector_labels_used_for_proposals') is not False or
            declared.get('original_waveform_copied') is not False or
            measured.get('reference_text_used_for_generation_or_selection') is not False or
            measured.get('new_http_requests') != 0 or measured.get('new_tts_requests') != 0 or
            measured.get('fresh_efficacy_sources') != 0 or measured.get('detector_checked') is not False or
            measured.get('independent_quality_evaluated') is not False):
        raise ValueError('Keep the original proposal limits and separation from reference evaluation')
    count = 0
    for actual, original, planned in zip(measured['cases'], generated['cases'], declared['cases']):
        for case in (actual, original, planned):
            if [p['donor'] for p in case['proposals']] != ['native-zero', 'continuous-local']:
                raise ValueError('Keep both generated donor slots in their original order')
        if any(actual[key] != original[key] or actual[key] != planned[key]
               for key in actual if key != 'proposals'):
            raise ValueError('A repair source or generated body changed')
        for current, proposal, plan in zip(actual['proposals'], original['proposals'], planned['proposals']):
            if any(current[key] != proposal[key] for key in proposal if key != 'status'):
                raise ValueError('A rendered proposal changed during assessment')
            if any(current[key] != plan[key] for key in plan if key != 'status'):
                raise ValueError('A donor changed after prospective declaration')
            if proposal['status'] == 'rendered_requires_speech_and_voice_checks':
                count += 1
                if current['status'] not in RENDERED or current.get('proposal_status') != proposal['status']:
                    raise ValueError('Every rendered output needs a completed processing decision')
            elif proposal['status'] not in UNRENDERED or current['status'] != proposal['status']:
                raise ValueError('Keep every unrendered proposal and its original reason')
    if count != measured['expected_rendered_outputs']:
        raise ValueError('Rendered outputs disappeared from the assessed inventory')


def data(generation_path, study_path):
    import soundfile as sf
    generation_path, study_path = Path(generation_path).resolve(), Path(study_path).resolve()
    directory = study_path.parent.parent; repair = directory/'ticodec_cached_repair_development'
    assessment = directory/'ticodec_cached_repair_assessment'
    if (generation_path != directory/'xcodec2_development/generation.json' or
            study_path != directory/'codec_profiles_fresh/curated_complete_report.json'):
        raise ValueError('Use the canonical complete panels and their original source study')
    public = xcodec_export(directory); generated = load(generation_path); study = load(study_path)
    expected = [c for c in study['cases'] if c['marked_provenance'] and not c['qualified_marked_transition']]
    repair_path = assessment/'assessment.json'; measured = load(repair_path)
    rendered, declared = load(repair/'generation.json'), load(repair/'declaration.json')
    validate_repair_inventory(measured, rendered, declared, [c['id'] for c in expected])
    if rendered['declaration_sha256'] != digest(repair/'declaration.json'):
        raise ValueError('The generated repair panel differs from its prospective declaration')
    code = dict(generated['code_files_sha256'])
    for name, sha in measured['code_files_sha256'].items():
        if name in code and code[name] != sha: raise ValueError('Generation implementations conflict')
        code[name] = sha
    for name in ('experiments/check_codec_followups.py', 'experiments/check_codec_frames.py',
                 'experiments/check_ticodec_latents.py', 'experiments/xcodec2_generation_report.py',
                 'experiments/codec_followup_panel_report.py', 'experiments/ticodec_panel_report.py',
                 'docs/codec-followup-endpoint.md'):
        code[str(ROOT/name)] = digest(ROOT/name)
    files = dict(measured['input_files_sha256'])
    files.update(generated['input_files_sha256'])
    files[str(repair_path)] = digest(repair_path)
    for name in ('evaluation.plan.json', 'original_evaluation.json', 'original_voice.json'):
        path = study_path.parent/name
        files[str(path)] = digest(path)
    if digest(study_path.parent/'evaluation.plan.json') != study['endpoint_receipts']['baseline']['plan_sha256']:
        raise ValueError('Keep the original authored references for subsequent independent scoring')
    for case in generated['cases']:
        cache = generation_path.parent/'recognition/processed'/(case['output_sha256']+'.json')
        if digest(cache) != case['recognition_cache_sha256']:
            raise ValueError('X-Codec2 recognition receipt changed')
        files[str(cache)] = digest(cache)
        if case.get('runtime_quality_guard') is not None:
            receipt = generation_path.parent/'runtime_voice'/case['id']/(case['output_sha256']+'.json')
            if digest(receipt) != case['runtime_quality_guard']['receipt_sha256']:
                raise ValueError('X-Codec2 voice receipt changed')
            files[str(receipt)] = digest(receipt)
    for name, sha in {**code, **files}.items():
        if digest(Path(name)) != sha: raise ValueError('A frozen input, implementation or receipt changed')
    known, history = previous_receipts(directory, study); files.update(history)
    voice = load(study_path.parent/'original_voice.json')
    identity = measured['recognition_configuration_sha256']
    if identity != generated['recognition_configuration_sha256']:
        raise ValueError('Generation panels used different processing recognizers')
    from codec_content import assess
    sources, treatments = [], []
    for old, xc, rc in zip(expected, public['cases'], measured['cases']):
        source = Path(rc['source']); info = sf.info(source)
        grid = {'samples': info.frames, 'sample_rate': info.samplerate, 'channels': info.channels}
        if (digest(source) != old['source_sha256'] or rc['source_sha256'] != old['source_sha256'] or
                grid != old['source_format'] or old['original_detected'] is not True or
                not 5 <= info.duration <= 300 or source.stat().st_size > 32_000_000):
            raise ValueError('Use unchanged complete marked sources within service limits')
        sources.append({'id': old['id'], 'file': str(source), 'sha256': old['source_sha256'], 'format': grid})
        xraw = next(c for c in generated['cases'] if c['id'] == old['id'])
        treatments.append({'id': old['id'], 'method': 'xcodec2-800bps', 'file': xraw['output_file'],
            'sha256': xc['output_sha256'], 'processing_status': xc['status'],
            'detector_eligible': xc['status'] == 'processing_eligible',
            'endpoint_phase_status': 'pending' if xc['status'] == 'processing_eligible' else 'processing_rejected'})
        before = read_recognition(study_path.parent/'recognition/originals', old['source_sha256'], identity)
        for proposal in rc['proposals']:
            status = proposal['status']
            row = {'id': old['id'], 'method': 'ticodec-repair-'+proposal['donor'],
                'processing_status': status, 'detector_eligible': status == 'processing_eligible',
                'endpoint_phase_status': 'pending' if status == 'processing_eligible' else
                                         'processing_rejected' if status in RENDERED else 'not_rendered'}
            if status in RENDERED:
                output = Path(proposal['output_file']); actual = sf.info(output)
                sha = proposal['output_sha256']; repair_info = proposal['repair']
                if (digest(output) != sha or not output.resolve().is_relative_to(repair) or
                        (actual.frames, actual.samplerate, actual.channels) !=
                        (info.frames, info.samplerate, info.channels) or output.stat().st_size > 32_000_000 or
                        proposal['source_format'] != grid or
                        not 0 < repair_info['repaired_fraction'] <= .20 or
                        not 1 <= len(repair_info['regions_seconds']) <= 8):
                    raise ValueError('Keep exact complete repairs within the original coverage limits')
                cache = assessment/'recognition/processed'/(sha+'.json')
                if digest(cache) != proposal['recognition_cache_sha256']:
                    raise ValueError('Repair recognition receipt changed')
                files[str(cache)] = digest(cache)
                speech = assess(before, read_recognition(cache.parent, sha, identity))
                if speech != proposal['speech_assessment']:
                    raise ValueError('Repair speech assessment differs from its original cache')
                quality = proposal.get('runtime_quality_guard')
                if status == 'content_rejected':
                    if speech['content_guard_passed'] or quality is not None:
                        raise ValueError('A speech rejection cannot imply a completed voice assessment')
                else:
                    receipt_path = assessment/'runtime_voice'/old['id']/(sha+'.json')
                    if digest(receipt_path) != quality['receipt_sha256']:
                        raise ValueError('Repair voice receipt changed')
                    files[str(receipt_path)] = digest(receipt_path)
                    receipt = load(receipt_path); clip = receipt['clips'][0]
                    original = next(c for c in voice['clips'] if c['id'] == old['id'])
                    passing = clip['speaker_minimum'] >= .95 and clip['dnsmos_delta']['overall'] >= -.2
                    if (not speech['content_guard_passed'] or receipt.get('complete') is not True or
                            len(receipt['clips']) != 1 or receipt['metric'] != voice['metric'] or
                            clip['original_dnsmos'] != original['original_dnsmos'] or
                            clip['source_sha256'] != old['source_sha256'] or clip['output_sha256'] != sha or
                            clip['speaker_minimum'] != quality['speaker_minimum'] or
                            clip['dnsmos_delta']['overall'] != quality['dnsmos_overall_delta'] or
                            quality['passed'] is not passing or passing != (status == 'processing_eligible') or
                            quality['speaker_minimum_required'] != .95 or quality['dnsmos_overall_delta_required'] != -.2):
                        raise ValueError('Repair eligibility differs from the original voice limits')
                row.update(file=str(output), sha256=sha)
            treatments.append(row)
    pending = [t for t in treatments if t['detector_eligible']]
    unique = len({t['sha256'] for t in pending})
    return {'kind': 'Declared spent-source X-Codec2 and generated-donor repair endpoint panel',
        'observation_report_kind': 'Spent-source X-Codec2 and generated-donor repair detector observations',
        'progress_prefix': 'CODEC_FOLLOWUP_DETECTOR', 'generation_file': str(generation_path),
        'generation_sha256': digest(generation_path), 'completed_study_file': str(study_path),
        'completed_study_sha256': digest(study_path), 'repair_assessment_sha256': digest(repair_path),
        'code_files_sha256': code, 'assessment_files_sha256': files, 'sources': sources, 'treatments': treatments,
        'historical_exact_file_observations': known, 'planned_sources': 25, 'planned_generation_treatments': 75,
        'operationally_eligible_treatments': len(pending), 'eligible_treatments': len(pending),
        'unique_eligible_files': unique, 'maximum_http_attempts': 2*unique+4,
        'minimum_detector_start_interval_seconds': 80, 'repetitions_per_exact_file': 2,
        'opening_source_id': sources[0]['id'], 'closing_source_id': sources[-1]['id'],
        'new_tts_requests': 0, 'fresh_efficacy_sources': 0,
        'reference_evaluation_order': 'After complete endpoint receipts; score every eligible treatment, including positives',
        'human_listening_approval': False, 'physical_erasure_proven': False}


def declare(generation_path, study_path, path):
    if path.exists(): raise ValueError('Never overwrite a declared endpoint inventory')
    value = data(generation_path, study_path); value['declared_at'] = endpoint.now()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        import json
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    first = sub.add_parser('declare'); first.add_argument('--generation', type=Path, required=True)
    first.add_argument('--completed-study', type=Path, required=True)
    first.add_argument('--declaration', type=Path, required=True)
    second = sub.add_parser('detect'); second.add_argument('--declaration', type=Path, required=True)
    second.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'declare':
        result = declare(args.generation, args.completed_study, args.declaration)
        print('CODEC_FOLLOWUPS_DECLARED', result['planned_generation_treatments'], 'rows;',
              result['unique_eligible_files'], 'unique files;', result['maximum_http_attempts'], 'HTTP maximum')
    else: endpoint.run(args.declaration, args.output, builder=data)
