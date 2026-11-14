"""Export metadata-only evidence from the frozen reconstruction studies.

    python experiments/reconstruction_reliability_report.py \
        --directory .cache/removal-reliability --output results/reconstruction_reliability.json

Use --study vocoder for the smoothed-vocoder study and fixed fallback workflow.

No API calls, audio exports, or transcript exports. Incomplete, duplicate, or
hash-inconsistent measurements fail instead of changing the denominator.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows, expected_ids, label):
    result = {}
    for row in rows:
        if row['id'] in result:
            raise ValueError(f'Duplicate {label}: {row["id"]}')
        result[row['id']] = row
    if set(result) != expected_ids:
        raise ValueError(f'Incomplete or unexpected {label} IDs')
    return result


def paired_counts(rows):
    positive = [row for row in rows if row['original_detected']]
    negative = [row for row in rows if not row['original_detected']]
    return {'recordings': len(rows), 'original_positive': len(positive),
            'positive_to_negative': sum(not row['processed_detected'] for row in positive),
            'remaining_positive': sum(row['processed_detected'] for row in positive),
            'original_negative': len(negative),
            'negative_to_positive': sum(row['processed_detected'] for row in negative),
            'negative_to_negative': sum(not row['processed_detected'] for row in negative)}


def content_metrics(record):
    fields = ('language', 'unit', 'original_units', 'processed_units', 'asr_disagreement',
              'reference_units', 'original_reference_error', 'processed_reference_error', 'reference_error_delta')
    return {key: record[key] for key in fields if key in record}


def verification_summary(audit, environment):
    if (audit['status'] != 'verified_negative' or audit['quality_verified'] is not False or
            audit['boundary_safe_requested'] is not True):
        raise ValueError('Exact-file verification smoke test did not pass')
    selected = [attempt for attempt in audit['attempts']
                if attempt.get('sha256') == audit['output_sha256'] and
                attempt.get('http_status') == 200 and attempt.get('detected') is False]
    if len(selected) != 1:
        raise ValueError('Smoke test has no unique checked output')
    return {'status': audit['status'], 'source_sha256': audit['source_sha256'],
            'output_sha256': audit['output_sha256'], 'selected_preset': audit['selected_preset'],
            'checked_at': audit['checked_at'], 'quality_verified': False,
            'format': 'FLOAT WAV', 'exact_bytes_rechecked_after_publication': True,
            'parent_environment': environment,
            'interpretation': 'Integration smoke test on a spent generated fixture; excluded from held-out and removal denominators.'}


def summarize_content(records):
    result = {}
    for unit in sorted({record['unit'] for record in records}):
        group = [record for record in records if record['unit'] == unit]
        deltas = [record['reference_error_delta'] for record in group if 'reference_error_delta' in record]
        result[unit] = {'recordings': len(group),
                        'with_reference_text': len(deltas),
                        'median_asr_disagreement': median(record['asr_disagreement'] for record in group),
                        'median_reference_error_delta': median(deltas) if deltas else None,
                        'maximum_reference_error_delta': max(deltas) if deltas else None}
    return result


def build_core_report(directory):
    plan_path = directory/'boundary_v2.plan.json'
    plan = json.loads(plan_path.read_text())
    plan_sha = digest(plan_path)
    paths = {key: directory/name for key, name in (
        ('api', 'boundary_v2.api.json'), ('render', 'boundary_v2.render.json'),
        ('asr', 'boundary_v2.asr.json'), ('secondary_asr', 'boundary_v2.asr_turbo.json'),
        ('voice', 'boundary_v2.voice_quality.json'))}
    documents = {key: json.loads(path.read_text()) for key, path in paths.items()}
    for name, key in (('api', 'measurement_complete'), ('asr', 'complete'),
                      ('secondary_asr', 'complete'), ('voice', 'complete')):
        if documents[name].get('plan_sha256') != plan_sha or documents[name].get(key) is not True:
            raise ValueError(f'{name} is incomplete or belongs to another frozen plan')
    items = indexed(plan['items'], {item['id'] for item in plan['items']}, 'plan')
    ids = set(items)
    renders = indexed(documents['render'], ids, 'render')
    asr = indexed(documents['asr']['clips'], ids, 'ASR')
    fresh_ids = {key for key, item in items.items() if item['cohort'] == 'fresh_native'}
    secondary_asr = indexed(documents['secondary_asr']['clips'], fresh_ids, 'secondary ASR')
    voice = indexed(documents['voice']['clips'], ids, 'voice quality')
    api = documents['api']
    if sum(r.get('status') == 200 for r in api['http_attempts']) != len(api['clips']):
        raise ValueError('HTTP success count differs from recorded endpoint responses')
    originals = indexed([r for r in api['clips'] if r['role'] == 'original'], ids, 'original API')
    processed = indexed([r for r in api['clips'] if r['role'] == 'processed'], ids, 'processed API')
    sentinels = [r for r in api['clips'] if r['role'] == 'original_sentinel']
    schedule = api['sentinel_schedule']['after_processed_counts']
    if sorted(r['sentinel_ordinal'] for r in sentinels) != schedule:
        raise ValueError('Missing or duplicate service sentinels')
    if len(api['clips']) != 2*len(items)+len(schedule):
        raise ValueError('Unexpected core endpoint observations')
    if len(api['http_attempts']) > plan['measurement']['maximum_http_attempts']:
        raise ValueError('HTTP attempt cap exceeded')
    if max(r['checked_at'] for r in originals.values()) >= min(r['checked_at'] for r in processed.values()):
        raise ValueError('Measurement order differs from the frozen protocol')
    rows = []
    for key, item in items.items():
        rendered = renders[key]
        source_sha, output_sha = item['sha256'], rendered['output_sha256']
        if rendered['source_sha256'] != source_sha or rendered['output'] != item['output']:
            raise ValueError(f'Mismatched render: {key}')
        if rendered['inference']['inference_implementation'] != 'reconstruction_inference.inference':
            raise ValueError(f'Wrong inference implementation: {key}')
        if (rendered['inference']['checkpoint_sha256'] != plan['model_sha256'] or
                rendered['inference']['settings'] != {k: v for k, v in plan['settings'].items() if k != 'seed'}):
            raise ValueError(f'Wrong model or settings: {key}')
        if (rendered['format_and_length_preserved'] is not True or rendered['finite_unclipped'] is not True or
                not 0 <= rendered['max_flac_quantization_error'] <= 2**-23):
            raise ValueError(f'Invalid recording or encoding: {key}')
        for record, expected in ((originals[key], source_sha), (processed[key], output_sha)):
            if (record['source_sha256'] != source_sha or record['queried_sha256'] != expected or
                    record['http_status'] != 200 or type(record['detected']) is not bool):
                raise ValueError(f'Invalid or mismatched API label: {key}')
        for record in (asr[key], voice[key]):
            if record['source_sha256'] != source_sha or record['output_sha256'] != output_sha:
                raise ValueError(f'Mismatched quality evidence: {key}')
        row = {field: item[field] for field in ('id', 'cohort', 'seconds', 'sample_rate', 'channels')}
        if 'marked_provenance' in item:
            row['declared_watermarked'] = item['marked_provenance']
        row['provenance_group'] = ('generated_marked' if item.get('marked_provenance') is True else
                                   'generated_unmarked' if item.get('marked_provenance') is False else
                                   'other_provider' if item.get('role') == 'other_provider_control' else
                                   'cartesia_published_model_unspecified')
        for field in ('model', 'language', 'voice_primary_language', 'url', 'page', 'role'):
            if field in item:
                row[field] = item[field]
        row.update(source_sha256=source_sha, output_sha256=output_sha,
                   original_detected=originals[key]['detected'], processed_detected=processed[key]['detected'],
                   original_checked_at=originals[key]['checked_at'], processed_checked_at=processed[key]['checked_at'],
                   max_encoding_error=rendered['max_flac_quantization_error'],
                   format_and_length_preserved=rendered['format_and_length_preserved'],
                   finite_unclipped=rendered['finite_unclipped'],
                   content=content_metrics(asr[key]),
                   voice={k: voice[key][k] for k in ('speaker_window_cosines', 'speaker_minimum', 'speaker_mean',
                          'original_dnsmos', 'processed_dnsmos', 'dnsmos_delta')})
        if key in secondary_asr:
            record = secondary_asr[key]
            if record['source_sha256'] != source_sha or record['output_sha256'] != output_sha:
                raise ValueError(f'Mismatched secondary recognizer evidence: {key}')
            row['secondary_content'] = content_metrics(record)
        rows.append(row)
    for record in sentinels:
        expected = originals[record['id']]
        if (record['id'] != api['sentinel_source_id'] or record['queried_sha256'] != expected['queried_sha256'] or
                record['http_status'] != 200 or type(record['detected']) is not bool):
            raise ValueError('Invalid service sentinel')
    counts = paired_counts(rows)
    fresh = [item for item in items.values() if item['cohort'] == 'fresh_native']
    return {'schema_version': 1, 'candidate': plan['candidate'], 'settings': plan['settings'],
            'model_sha256': plan['model_sha256'], 'inference_code_sha256': plan['code_sha256'],
            'plan_sha256': plan_sha, 'evidence_sha256': {key: digest(path) for key, path in paths.items()},
            'study_completed_at': api['completed_at'], 'exported_at': datetime.now(timezone.utc).isoformat(),
            'selection': {'public': 'All 5–300 second public page samples; full recordings, hash order',
                          'fresh': '16 public voice IDs absent from prior corpus plans/results, 20 authored utterances; four matched sonic-3.5 controls',
                          'fresh_voice_ids': len({item['voice'] for item in fresh}),
                          'fresh_languages': sorted({item['language'] for item in fresh}),
                          'fresh_sample_rates': sorted({item['sample_rate'] for item in fresh}),
                          'public_after_labels_unseen_before_fix': plan['public_after_labels_seen_before_fix'] is False},
            'measurement': {**plan['measurement'], 'actual_http_attempts': len(api['http_attempts']),
                            'http_status_counts': dict(Counter(str(r.get('status', 'no response')) for r in api['http_attempts'])),
                            'final_interval_seconds': api['current_interval_seconds'],
                            'sentinel_rechecks': len(sentinels),
                            'sentinels_remained_positive': all(r['detected'] for r in sentinels),
                            'execution_notes': api.get('execution_notes', [])},
            'summary': counts, 'by_cohort': {cohort: paired_counts([r for r in rows if r['cohort'] == cohort])
                                            for cohort in sorted({r['cohort'] for r in rows})},
            'by_provenance': {group: paired_counts([r for r in rows if r['provenance_group'] == group])
                              for group in sorted({r['provenance_group'] for r in rows})},
            'quality_metrics': {'asr': documents['asr']['metric'], 'speaker': documents['voice']['speaker_metric'],
                                'secondary_asr': documents['secondary_asr']['metric'],
                                'dnsmos': documents['voice']['dnsmos_metric'],
                                'content_by_unit': summarize_content([r['content'] for r in rows]),
                                'secondary_content_by_unit': summarize_content([r['secondary_content'] for r in rows
                                                                                 if 'secondary_content' in r]),
                                'minimum_speaker_window_cosine': min(r['voice']['speaker_minimum'] for r in rows),
                                'median_dnsmos_overall_delta': median(r['voice']['dnsmos_delta']['overall'] for r in rows),
                                'human_listening_verified': False},
            'interpretation': 'Observed complete-file endpoint responses for this frozen candidate. Originally negative recordings do not count as removals. Sentinels are repeated controls, not new samples. No universal success, physical erasure, or human-rated speech fidelity claim.',
            'clips': rows}


def checked(record, expected_sha):
    if record['http_status'] != 200 or type(record['detected']) is not bool or record['sha256'] != expected_sha:
        raise ValueError('Invalid or mismatched follow-up endpoint observation')


def build_vocoder_component(directory, stem, plan_name):
    paths = {key: directory/name for key, name in (
        ('plan', plan_name), ('api', stem+'.api.json'), ('render', stem+'.render.json'),
        ('asr', stem+'.asr.json'), ('secondary_asr', stem+'.asr_turbo.json'),
        ('voice', stem+'.voice_quality.json'))}
    data = {key: json.loads(path.read_text()) for key, path in paths.items()}
    plan, api = data['plan'], data['api']
    plan_sha = digest(paths['plan'])
    if plan['after_labels_seen_before_seal'] is not False:
        raise ValueError('Vocoder recipe was not sealed before processed labels')
    for key, flag in (('api', 'measurement_complete'), ('asr', 'complete'),
                       ('secondary_asr', 'complete'), ('voice', 'complete')):
        if data[key].get(flag) is not True or data[key]['plan_sha256'] != plan_sha:
            raise ValueError(f'Incomplete or mismatched {stem} {key}')
    if api['candidate'] != plan['candidate'] or api['render_report_sha256'] != digest(paths['render']):
        raise ValueError('Vocoder candidate or render evidence changed')
    items = indexed(plan['items'], {i['id'] for i in plan['items']}, 'vocoder plan')
    ids = set(items)
    renders = indexed(data['render'], ids, 'vocoder render')
    metrics = {key: indexed(data[key]['clips'], ids, 'vocoder '+key)
               for key in ('asr', 'secondary_asr', 'voice')}
    originals = indexed([r for r in api['clips'] if r['role'] == 'original'], ids, 'vocoder originals')
    processed = indexed([r for r in api['clips'] if r['role'] == 'processed'], ids, 'vocoder processed')
    repeat_ids = set(plan['measurement']['repeat_ids'])
    repeated = indexed([r for r in api['clips'] if r['role'] == 'processed_repeat'], repeat_ids, 'vocoder repeats')
    controls = [r for r in api['clips'] if r['role'] == 'original_sentinel']
    expected_controls = [0, len(ids), len(ids)+len(repeat_ids)]
    if (sorted(r['ordinal'] for r in controls) != expected_controls or
            len(api['clips']) != 2*len(ids)+len(repeat_ids)+len(expected_controls) or
            len(api['http_attempts']) > plan['measurement']['maximum_http_attempts'] or
            sum(r.get('status') == 200 for r in api['http_attempts']) != len(api['clips'])):
        raise ValueError('Invalid vocoder HTTP, repeat or service-control accounting')
    if (max(r['checked_at'] for r in originals.values()) >= min(r['checked_at'] for r in processed.values()) or
            max(r['checked_at'] for r in processed.values()) >= min(r['checked_at'] for r in repeated.values())):
        raise ValueError('Vocoder observation order differs from the fixed protocol')
    for row in controls:
        if row['id'] != api['sentinel_source_id']:
            raise ValueError('Wrong vocoder service control')
        checked(row, items[row['id']]['sha256'])
    rows = []
    for key, item in items.items():
        render = renders[key]
        source_sha, output_sha = item['sha256'], render['output_sha256']
        checked(originals[key], source_sha)
        checked(processed[key], output_sha)
        if key in repeated:
            checked(repeated[key], output_sha)
        if (render['source_sha256'] != source_sha or render['output'] != item['output'] or
                render['format_and_length_preserved'] is not True or render['finite_unclipped'] is not True or
                not 0 <= render['max_flac_quantization_error'] <= 2**-23 or
                render['inference']['settings'] != plan['settings'] or
                render['inference']['checkpoint_sha256'] != plan['candidate']['model_sha256'] or
                render['inference']['source_commit'] != plan['candidate']['source_commit']):
            raise ValueError(f'Invalid vocoder reconstruction: {key}')
        for group in metrics.values():
            if group[key]['source_sha256'] != source_sha or group[key]['output_sha256'] != output_sha:
                raise ValueError(f'Mismatched vocoder quality evidence: {key}')
        voice = metrics['voice'][key]
        row = {field: item[field] for field in
               ('id', 'cohort', 'model', 'language', 'voice_primary_language', 'seconds', 'sample_rate', 'channels')}
        row.update(declared_watermarked=item['marked_provenance'],
                   provenance_group='generated_marked' if item['marked_provenance'] else 'generated_unmarked',
                   source_sha256=source_sha, output_sha256=output_sha,
                   original_detected=originals[key]['detected'], processed_detected=processed[key]['detected'],
                   original_checked_at=originals[key]['checked_at'], processed_checked_at=processed[key]['checked_at'],
                   max_encoding_error=render['max_flac_quantization_error'],
                   format_and_length_preserved=True, finite_unclipped=True,
                   content=content_metrics(metrics['asr'][key]), secondary_content=content_metrics(metrics['secondary_asr'][key]),
                   voice={field: voice[field] for field in ('speaker_window_cosines', 'speaker_minimum', 'speaker_mean',
                                                          'original_dnsmos', 'processed_dnsmos', 'dnsmos_delta')})
        if key in repeated:
            row.update(processed_repeat_detected=repeated[key]['detected'], repeat_checked_at=repeated[key]['checked_at'])
        rows.append(row)
    return {'plan_sha256': plan_sha, 'evidence_sha256': {key: digest(path) for key, path in paths.items()},
            'candidate': plan['candidate'], 'settings': plan['settings'], 'code_sha256': plan['code_sha256'],
            'summary': paired_counts(rows), 'clips': rows, 'http_attempts': len(api['http_attempts']),
            'http_status_counts': dict(Counter(str(r.get('status', 'no response')) for r in api['http_attempts'])),
            'repeated_outputs': len(repeated), 'repeated_outputs_negative': sum(not r['detected'] for r in repeated.values()),
            'positive_service_controls': sum(r['detected'] for r in controls), 'service_controls': len(controls),
            'completed_at': api['completed_at'],
            'metrics': {'asr': data['asr']['metric'], 'secondary_asr': data['secondary_asr']['metric'],
                        'speaker': data['voice']['speaker_metric'], 'dnsmos': data['voice']['dnsmos_metric']}}


def build_vocoder_report(directory):
    fresh = build_vocoder_component(directory, 'vocoder_v1', 'vocoder_eval.plan.json')
    long = build_vocoder_component(directory, 'vocoder_long', 'vocoder_long_eval.plan.json')
    if fresh['candidate'] != long['candidate'] or fresh['code_sha256'] != long['code_sha256']:
        raise ValueError('Long vocoder supplement changed the frozen candidate')
    source = json.loads((directory/'vocoder_source_commit.json').read_text())
    if source['files_sha256'] != fresh['code_sha256']:
        raise ValueError('Published vocoder source differs from the frozen study')
    rows = fresh['clips']+long['clips']
    ids = {r['id'] for r in rows}
    if len(ids) != len(rows):
        raise ValueError('Duplicate source recording across vocoder studies')
    workflow_path = directory/'vocoder_workflow.api.json'
    workflow_plan_path = directory/'vocoder_workflow.plan.json'
    workflow, protocol = (json.loads(p.read_text()) for p in (workflow_path, workflow_plan_path))
    if (workflow['measurement_complete'] is not True or workflow['plan_sha256'] != digest(workflow_plan_path) or
            protocol['sealed_before_primary_processed_labels'] is not True or
            protocol['primary_plan_sha256'] != fresh['plan_sha256'] or
            protocol['supplement_plan_sha256'] != long['plan_sha256'] or
            workflow['base_reports_sha256'] != {stem: digest(directory/(stem+'.api.json'))
                                               for stem in ('vocoder_v1', 'vocoder_long')} or
            len(workflow['http_attempts']) > protocol['maximum_fallback_http_attempts']):
        raise ValueError('Invalid or incomplete fixed fallback workflow')
    cases = indexed(workflow['cases'], ids, 'workflow outcomes')
    fallback_count = sum(case['fallback_needed'] for case in cases.values())
    labels = sorted(row['label'] for row in workflow['service_controls'])
    if labels != (['after_fallback', 'before_fallback'] if fallback_count else []):
        raise ValueError('Missing or unexpected fallback service controls')
    main_plan = json.loads((directory/'vocoder_eval.plan.json').read_text())
    main_api = json.loads((directory/'vocoder_v1.api.json').read_text())
    sentinel_sha = next(item['sha256'] for item in main_plan['items'] if item['id'] == main_api['sentinel_source_id'])
    for row in workflow['service_controls']:
        checked(row, sentinel_sha)
    final = []
    for row in rows:
        case = cases[row['id']]
        needed = row['processed_detected'] or row.get('processed_repeat_detected', False)
        if (case['source_sha256'] != row['source_sha256'] or case['fallback_needed'] != needed or
                case['primary_detected'] != row['processed_detected']):
            raise ValueError('Workflow selected from a different primary observation')
        if case['status'] == 'verified_negative':
            if digest(Path(__file__).resolve().parents[1]/case['output']) != case['output_sha256']:
                raise ValueError('Selected workflow file changed after verification')
            if not needed and (case['selected_preset'] != 'smoothed-vocoder' or case['output_sha256'] != row['output_sha256']):
                raise ValueError('Workflow silently replaced a primary negative file')
            if needed:
                audit_path = directory/'vocoder_v1/fallback'/(row['id']+'.audit.json')
                audit = json.loads(audit_path.read_text())
                if (digest(audit_path) != case['audit_sha256'] or audit['status'] != 'verified_negative' or
                        audit['output_sha256'] != case['output_sha256'] or
                        audit['selected_preset'] not in protocol['fallback_presets']):
                    raise ValueError('Fallback output lacks its exact-file audit')
        final.append({key: case[key] for key in ('id', 'source_sha256', 'primary_detected', 'primary_repeat_detected',
                                                'fallback_needed', 'status', 'quality_verified') } |
                     {key: case[key] for key in ('selected_preset', 'output_sha256', 'checked_at') if key in case})
    primary_plan = json.loads((directory/'vocoder_eval.plan.json').read_text())
    long_plan = json.loads((directory/'vocoder_long_eval.plan.json').read_text())
    reservations = json.loads((directory/'budget.json').read_text())['requests']
    generated_ids = {i['id'] for i in primary_plan['items']}
    generated = indexed([r for r in reservations if r['id'] in generated_ids], generated_ids, 'vocoder generation')
    for item in primary_plan['items']:
        if generated[item['id']]['status'] != 'completed' or generated[item['id']]['output_sha256'] != item['sha256']:
            raise ValueError('Fresh source lacks a completed generation receipt')
    regressions = json.loads((directory/'vocoder_regressions.api.json').read_text())
    prototype = json.loads((directory/'smooth_bigvgan.api.json').read_text())
    if len(regressions['clips']) != 2 or len(prototype['clips']) != 2:
        raise ValueError('Development regression checks are incomplete')
    for row in regressions['clips']+prototype['clips']:
        if row['http_status'] != 200 or type(row['detected']) is not bool:
            raise ValueError('Invalid development regression observation')
    return {'schema_version': 1, 'candidate': fresh['candidate'], 'settings': fresh['settings'],
            'inference_source_commit': source['commit'], 'inference_code_sha256': fresh['code_sha256'],
            'upstream_model_trained_by_this_project': False,
            'selection': {'fresh_recordings': len(fresh['clips']),
                          'fresh_voice_ids': len({i['voice'] for i in primary_plan['items']}),
                          'fresh_languages': sorted({i['language'] for i in primary_plan['items']}),
                          'english_voice_note': 'Fresh English uses a previously unused German-primary multilingual voice with English requested.',
                          'long_supplement': long_plan['selection'],
                          'no_universal_or_independent_population_claim': True},
            'summary': paired_counts(rows),
            'by_provenance': {group: paired_counts([r for r in rows if r['provenance_group'] == group])
                              for group in sorted({r['provenance_group'] for r in rows})},
            'fresh_study': {key: value for key, value in fresh.items() if key not in ('clips', 'metrics')},
            'long_study': {key: value for key, value in long.items() if key not in ('clips', 'metrics')},
            'quality_metrics': {**fresh['metrics'],
                                'content_by_unit': summarize_content([r['content'] for r in rows]),
                                'secondary_content_by_unit': summarize_content([r['secondary_content'] for r in rows]),
                                'minimum_speaker_window_cosine': min(r['voice']['speaker_minimum'] for r in rows),
                                'median_dnsmos_overall_delta': median(r['voice']['dnsmos_delta']['overall'] for r in rows),
                                'human_listening_verified': False},
            'development_regressions': {'recordings': 4,
                                        'negative_responses': sum(not r['detected'] for r in regressions['clips']+prototype['clips']),
                                        'not_in_held_out_denominator': True,
                                        'individual_private_fingerprints_excluded': True},
            'official_source_parity': json.loads((directory/'bigvgan_source_parity.json').read_text()),
            'native_generation': {'new_requests': len(generated),
                                   'characters': sum(r['characters'] for r in generated.values()),
                                   'reused_long_sources': len(long['clips'])},
            'fixed_fallback_workflow': {'plan_sha256': digest(workflow_plan_path), 'report_sha256': digest(workflow_path),
                                         'decision_rule': protocol['decision_rule'], 'repeat_handling': protocol['repeat_handling'],
                                         'source_recordings': len(cases), 'fallback_sources': sum(r['fallback_needed'] for r in cases.values()),
                                         'verified_negative_sources': sum(r['status'] == 'verified_negative' for r in cases.values()),
                                         'http_attempts': len(workflow['http_attempts']), 'outcomes': final,
                                         'human_listening_verified': False,
                                         'interpretation': 'Same source recordings; fallback queries do not increase the sample count. Speech metrics above describe the primary candidate, not any selected fallback.'},
            'environment': {name: json.loads((directory/f'{name}_environment.json').read_text())
                             for name in ('vocoder', 'asr', 'voice')},
            'all_primary_processed_responses_negative': all(not r['processed_detected'] and not r.get('processed_repeat_detected', False) for r in rows),
            'service_controls_all_positive': all(c['positive_service_controls'] == c['service_controls'] for c in (fresh, long)) and
                                               all(r['detected'] for r in workflow['service_controls']),
            'interpretation': 'Frozen whole-file measurements. Controls, prior development cases, and repeated queries do not inflate removal counts. No claim of universal erasure, unchanged words, or later endpoint behavior.',
            'clips': rows, 'study_completed_at': workflow['completed_at'], 'exported_at': datetime.now(timezone.utc).isoformat()}


def build_report(directory):
    report = build_core_report(directory)
    source = json.loads((directory/'frozen_source_commit.json').read_text())
    if source['files_sha256'] != report['inference_code_sha256']:
        raise ValueError('Frozen source commit has different inference files')
    report['inference_source_commit'] = source['commit']
    audit = json.loads((directory/'verified_current_atomic.audit.json').read_text())
    expected = {Path(name).name: sha for name, sha in report['inference_code_sha256'].items()}
    if audit['source_files_sha256_at_start'] != expected or digest(Path(audit['output'])) != audit['output_sha256']:
        raise ValueError('Verification smoke test code or published file changed')
    report['verification_cli_smoke'] = verification_summary(
        audit, json.loads((directory/'verification_environment.json').read_text()))
    core_rows = {row['id']: row for row in report['clips']}
    core_plan = json.loads((directory/'boundary_v2.plan.json').read_text())
    core_items = {item['id']: item for item in core_plan['items']}
    core_api = json.loads((directory/'boundary_v2.api.json').read_text())

    repeat_plan_path = directory/'boundary_v2.repeat.plan.json'
    repeat_path = directory/'boundary_v2.repeat.api.json'
    repeat_plan, repeat = (json.loads(path.read_text()) for path in (repeat_plan_path, repeat_path))
    if (repeat.get('measurement_complete') is not True or repeat['plan_sha256'] != digest(repeat_plan_path) or
            repeat['core_plan_sha256'] != report['plan_sha256'] or
            repeat['core_api_sha256'] != digest(directory/'boundary_v2.api.json')):
        raise ValueError('Incomplete or mismatched exact-file repetitions')
    targets = {item['id']: item for item in repeat_plan['items']}
    repeated = indexed([row for row in repeat['clips'] if row['role'] == 'processed_repeat'], set(targets), 'repeat')
    sentinels = [row for row in repeat['clips'] if row['role'] == 'original_sentinel']
    if (len(repeat['clips']) != len(targets)+len(repeat_plan['sentinel_after_target_counts']) or
            sorted(row['ordinal'] for row in sentinels) != repeat_plan['sentinel_after_target_counts'] or
            len(repeat['http_attempts']) > repeat_plan['maximum_http_attempts'] or
            sum(row.get('status') == 200 for row in repeat['http_attempts']) != len(repeat['clips'])):
        raise ValueError('Invalid repetition or service-control accounting')
    sentinel_id = repeat['sentinel_source_id']
    for row in sentinels:
        if row['id'] != sentinel_id:
            raise ValueError('Wrong repeat service sentinel')
        checked(row, core_items[sentinel_id]['sha256'])
    public_repeats, private_repeats = [], []
    for key, item in targets.items():
        expected = core_rows[key]['output_sha256'] if item['render_bound'] else item['sha256']
        row = repeated[key]
        checked(row, expected)
        if item['render_bound']:
            core_rows[key]['processed_repeat_detected'] = row['detected']
            public_repeats.append({field: row[field] for field in ('id', 'sha256', 'detected', 'http_status', 'checked_at')})
        else:
            private_repeats.append(row)
    report['repeat_stability'] = {
        'plan_sha256': digest(repeat_plan_path), 'report_sha256': digest(repeat_path),
        'fixed_targets': len(targets), 'negative_responses': sum(not row['detected'] for row in repeated.values()),
        'changed_public_responses': [key for key in targets if key in core_rows and
                                     repeated[key]['detected'] != core_rows[key]['processed_detected']],
        'http_attempts': len(repeat['http_attempts']), 'positive_service_controls': sum(row['detected'] for row in sentinels),
        'service_controls': len(sentinels), 'public_observations': public_repeats,
        'execution_notes': repeat.get('execution_notes', []),
        'private_development_regressions': {'recordings': len(private_repeats),
                                            'negative_responses': sum(not row['detected'] for row in private_repeats)},
        'interpretation': 'Repeated exact-file responses are not additional independent recordings. Private development inputs and their individual fingerprints remain local.'}

    paths = {key: directory/name for key, name in (
        ('plan', 'long_eval.plan.json'), ('render', 'long_v2.render.json'), ('api', 'long_v2.api.json'),
        ('asr', 'long_v2.asr.json'), ('secondary_asr', 'long_v2.asr_turbo.json'), ('voice', 'long_v2.voice_quality.json'))}
    long = {key: json.loads(path.read_text()) for key, path in paths.items()}
    long_plan_sha = digest(paths['plan'])
    for name, flag in (('api', 'measurement_complete'), ('asr', 'complete'), ('secondary_asr', 'complete'), ('voice', 'complete')):
        if long[name].get(flag) is not True or long[name]['plan_sha256'] != long_plan_sha:
            raise ValueError(f'Incomplete or mismatched long {name} report')
    if long['plan']['code_sha256'] != report['inference_code_sha256'] or long['plan']['settings'] != report['settings']:
        raise ValueError('Long study changed the frozen candidate')
    items = indexed(long['plan']['items'], {item['id'] for item in long['plan']['items']}, 'long plan')
    ids = set(items)
    renders = indexed(long['render'], ids, 'long render')
    asr = indexed(long['asr']['clips'], ids, 'long ASR')
    secondary = indexed(long['secondary_asr']['clips'], ids, 'long secondary ASR')
    voice = indexed(long['voice']['clips'], ids, 'long voice quality')
    api = long['api']
    if (api['render_report_sha256'] != digest(paths['render']) or api['core_repeat_report_sha256'] != digest(repeat_path) or
            len(api['http_attempts']) > long['plan']['measurement']['maximum_http_attempts'] or
            sum(row.get('status') == 200 for row in api['http_attempts']) != len(api['clips'])):
        raise ValueError('Long measurement hashes or HTTP accounting differ')
    originals = indexed([row for row in api['clips'] if row['role'] == 'original'], ids, 'long originals')
    processed = indexed([row for row in api['clips'] if row['role'] == 'processed'], ids, 'long processed')
    repetitions = indexed([row for row in api['clips'] if row['role'] == 'processed_repeat'], ids, 'long repeats')
    long_sentinels = [row for row in api['clips'] if row['role'] == 'original_sentinel']
    if (len(api['clips']) != 3*len(items)+len(long['plan']['measurement']['sentinel_after_processed_counts']) or
            sorted(row['ordinal'] for row in long_sentinels) != long['plan']['measurement']['sentinel_after_processed_counts']):
        raise ValueError('Long observations or service sentinels are incomplete')
    for row in long_sentinels:
        if row['id'] != api['sentinel_source_id']:
            raise ValueError('Wrong long service sentinel')
        checked(row, core_items[row['id']]['sha256'])
    long_rows = []
    for key, item in items.items():
        rendered = renders[key]
        source_sha, output_sha = item['sha256'], rendered['output_sha256']
        checked(originals[key], source_sha)
        checked(processed[key], output_sha)
        checked(repetitions[key], output_sha)
        if (rendered['source_sha256'] != source_sha or rendered['output'] != item['output'] or
                rendered['format_and_length_preserved'] is not True or rendered['finite_unclipped'] is not True or
                not 0 <= rendered['max_flac_quantization_error'] <= 2**-23 or
                rendered['inference']['checkpoint_sha256'] != report['model_sha256'] or
                rendered['inference']['inference_implementation'] != 'reconstruction_inference.inference' or
                rendered['inference']['settings'] != {k: v for k, v in report['settings'].items() if k != 'seed'}):
            raise ValueError(f'Invalid or mismatched long reconstruction: {key}')
        for metric in (asr[key], secondary[key], voice[key]):
            if metric['source_sha256'] != source_sha or metric['output_sha256'] != output_sha:
                raise ValueError(f'Mismatched long quality evidence: {key}')
        row = {field: item[field] for field in ('id', 'cohort', 'model', 'language', 'seconds', 'sample_rate', 'channels',
                                               'voice_primary_language', 'voice_absent_from_prior_project_plans')}
        row.update(declared_watermarked=item['marked_provenance'],
                   provenance_group='generated_marked' if item['marked_provenance'] else 'generated_unmarked',
                   source_sha256=source_sha, output_sha256=output_sha,
                   original_detected=originals[key]['detected'], processed_detected=processed[key]['detected'],
                   processed_repeat_detected=repetitions[key]['detected'],
                   original_checked_at=originals[key]['checked_at'], processed_checked_at=processed[key]['checked_at'],
                   repeat_checked_at=repetitions[key]['checked_at'],
                   max_encoding_error=rendered['max_flac_quantization_error'],
                   format_and_length_preserved=True, finite_unclipped=True,
                   content=content_metrics(asr[key]), secondary_content=content_metrics(secondary[key]),
                   voice={field: voice[key][field] for field in ('speaker_window_cosines', 'speaker_minimum', 'speaker_mean',
                                                               'original_dnsmos', 'processed_dnsmos', 'dnsmos_delta')})
        long_rows.append(row)
    report['core_study_summary'] = report['summary']
    report['long_study'] = {'plan_sha256': long_plan_sha, 'evidence_sha256': {key: digest(path) for key, path in paths.items()},
                            'summary': paired_counts(long_rows), 'duration_seconds': [row['seconds'] for row in long_rows],
                            'distinct_voices': len({item['voice'] for item in items.values()}),
                            'correlation_note': long['plan']['correlation_note'], 'http_attempts': len(api['http_attempts']),
                            'repeated_outputs_negative': sum(not row['detected'] for row in repetitions.values()),
                            'repeated_outputs': len(repetitions), 'positive_service_controls': sum(row['detected'] for row in long_sentinels),
                            'service_controls': len(long_sentinels),
                            'execution_notes': [json.loads((directory/'long_v2.scheduling_note.json').read_text())]}
    report['clips'].extend(long_rows)
    rows = report['clips']
    report['summary'] = paired_counts(rows)
    report['by_cohort'] = {group: paired_counts([row for row in rows if row['cohort'] == group])
                           for group in sorted({row['cohort'] for row in rows})}
    report['by_provenance'] = {group: paired_counts([row for row in rows if row['provenance_group'] == group])
                              for group in sorted({row['provenance_group'] for row in rows})}
    report['quality_metrics'].update(
        content_by_unit=summarize_content([row['content'] for row in rows]),
        secondary_content_by_unit=summarize_content([row['secondary_content'] for row in rows if 'secondary_content' in row]),
        minimum_speaker_window_cosine=min(row['voice']['speaker_minimum'] for row in rows),
        median_dnsmos_overall_delta=median(row['voice']['dnsmos_delta']['overall'] for row in rows))
    report['environment'] = {name: json.loads((directory/f'{name}_environment.json').read_text()) for name in ('asr', 'voice')}
    budget = json.loads((directory/'budget.json').read_text())
    if budget['reserved_credits'] > budget['maximum_reserved_credits']:
        raise ValueError('Generation spending reservation exceeded its cap')
    native_ids = {row['id'] for row in rows if row['cohort'] in ('fresh_native', 'long_native')}
    reservations = indexed([row for row in budget['requests'] if row['id'] in native_ids],
                           native_ids, 'native generation reservation')
    report['native_generation'] = {'requests': len(reservations),
                                   'characters': sum(row['characters'] for row in reservations.values()),
                                   'utf8_bytes': sum(row['utf8_bytes'] for row in reservations.values())}
    correction = json.loads((directory/'asr_padding_revision.json').read_text())
    report['evaluation_corrections'] = [{key: correction[key] for key in
                                         ('at', 'scope', 'bug', 'correction', 'cache_policy')}]
    report['quality_protocol_sha256'] = {name: digest(directory/name) for name in
                                         ('boundary_v2.quality_protocol.json', 'long_v2.quality_protocol.json')}
    report['total_study_http_attempts'] = (report['measurement']['actual_http_attempts']+
                                           len(repeat['http_attempts'])+len(api['http_attempts']))
    report['all_processed_responses_negative'] = (all(not row['processed_detected'] for row in rows) and
                                                  all(not row['detected'] for row in repeated.values()) and
                                                  all(not row['detected'] for row in repetitions.values()))
    report['service_controls_all_positive'] = (report['measurement']['sentinels_remained_positive'] and
                                               all(row['detected'] for row in sentinels+long_sentinels))
    report['study_completed_at'] = api['completed_at']
    report['exported_at'] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=('full-flow', 'vocoder'), default='full-flow')
    args = parser.parse_args()
    report = build_report(args.directory) if args.study == 'full-flow' else build_vocoder_report(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
