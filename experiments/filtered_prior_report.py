"""Export metadata-only evidence for the frozen filtered-prior study.

    python experiments/filtered_prior_report.py --directory .cache/removal-reliability \
        --output results/reconstruction_filtered_prior.json

No audio, private transcripts, or individual private fingerprints are exported.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reconstruction_reliability_report import (
    build_vocoder_component, checked, content_metrics, digest, estimated_quality_gate,
    indexed, load_selected_fallback_quality, paired_counts, quality_summary,
    selected_quality_rows, summarize_content, verification_summary,
)


def load(path):
    return json.loads(path.read_text())


def public_workflow(directory, primary):
    path = directory/'filtered_workflow.api.json'
    protocol_path = directory/'filtered_workflow.plan.json'
    workflow, protocol = load(path), load(protocol_path)
    if (workflow['measurement_complete'] is not True or workflow['plan_sha256'] != digest(protocol_path) or
            protocol['sealed_before_primary_processed_labels'] is not True or
            protocol['required_fallback_negative_checks'] != 2 or
            protocol['fallback_presets'] != ['full-flow', 'strong'] or
            protocol['code_sha256'] != load(directory/'filtered_eval.plan.json')['code_sha256'] or
            protocol['primary_plan_sha256'] != digest(directory/'filtered_eval.plan.json') or
            protocol['supplement_plan_sha256'] != digest(directory/'filtered_long_eval.plan.json') or
            workflow['base_reports_sha256'] != {stem: digest(directory/(stem+'.api.json'))
                                               for stem in ('filtered_v1', 'filtered_long')} or
            len(workflow['http_attempts']) > protocol['maximum_fallback_http_attempts']):
        raise ValueError('Incomplete or inconsistent filtered-prior workflow')
    cases = indexed(workflow['cases'], {r['id'] for r in primary}, 'selected workflow')
    needed = sum(c['fallback_needed'] for c in cases.values())
    if sorted(c['label'] for c in workflow['service_controls']) != (['after_fallback', 'before_fallback'] if needed else []):
        raise ValueError('Missing fallback service controls')
    main_api = load(directory/'filtered_v1.api.json')
    sentinel = next(i for i in load(directory/'filtered_eval.plan.json')['items']
                    if i['id'] == main_api['sentinel_source_id'])
    for control in workflow['service_controls']:
        checked(control, sentinel['sha256'])
    exported = []
    for original in primary:
        case = cases[original['id']]
        requires_fallback = original['processed_detected'] or original['processed_repeat_detected']
        if (case['source_sha256'] != original['source_sha256'] or
                case['fallback_needed'] != requires_fallback or
                case['primary_detected'] != original['processed_detected'] or
                case['primary_repeat_detected'] != original['processed_repeat_detected']):
            raise ValueError('Selection used different primary observations')
        if case['status'] == 'verified_negative':
            if digest(ROOT/case['output']) != case['output_sha256'] or case['negative_checks'] != 2:
                raise ValueError('Selected file changed or lacks two negative checks')
            if not requires_fallback:
                if case['selected_preset'] != 'filtered-prior' or case['output_sha256'] != original['output_sha256']:
                    raise ValueError('A primary negative file was silently replaced')
            else:
                audit_path = directory/'filtered_v1/fallback'/(case['id']+'.audit.json')
                audit = load(audit_path)
                verification_summary(audit, {})
                if (digest(audit_path) != case['audit_sha256'] or audit['source_sha256'] != case['source_sha256'] or
                        audit['output_sha256'] != case['output_sha256'] or
                        audit['required_negative_checks'] != 2 or
                        audit['selected_preset'] not in protocol['fallback_presets'] or
                        audit['verification_cli_code_sha256'] != protocol['code_sha256']['tools/verify_reconstruction.py']):
                    raise ValueError('Selected fallback lacks a matching two-check audit')
        exported.append({k: case[k] for k in ('id', 'source_sha256', 'primary_detected', 'primary_repeat_detected',
                                               'fallback_needed', 'status', 'negative_checks', 'quality_verified')} |
                        {k: case[k] for k in ('selected_preset', 'output_sha256', 'checked_at') if k in case})
    fallback, evidence = load_selected_fallback_quality(directory, 'filtered_fallback', cases, path)
    selected = selected_quality_rows(primary, cases, fallback)
    return {'plan_sha256': digest(protocol_path), 'report_sha256': digest(path),
            'decision_rule': protocol['decision_rule'], 'source_recordings': len(cases),
            'fallback_sources': needed, 'verified_negative_sources': len(selected),
            'all_sources_received_checked_negative': len(selected) == len(primary),
            'required_negative_checks': 2, 'http_attempts': len(workflow['http_attempts']),
            'service_controls': len(workflow['service_controls']),
            'positive_service_controls': sum(c['detected'] for c in workflow['service_controls']),
            'selected_output_quality': quality_summary(selected),
            'selected_quality_failures': [{'id': r['id'], 'criteria': estimated_quality_gate(r)}
                                           for r in selected if not all(estimated_quality_gate(r).values())],
            'fallback_quality_evidence_sha256': evidence, 'fallback_measurements': fallback,
            'outcomes': exported, 'completed_at': workflow['completed_at'],
            'interpretation': 'Same sixteen sources; fallback attempts and repeat queries add no recordings.'}


def regression_summary(primary, selected, original_labels, primary_labels, repeat_labels, cases, private_count):
    """Summarize private rows without exporting their IDs, hashes or content."""
    def descriptive_quality(rows):
        summary = quality_summary(rows)
        # No known source transcript exists for private uploads. Paired ASR
        # disagreement is reported; a missing reference is not a measured failure.
        summary.pop('estimated_criteria')
        summary.pop('recordings_meeting_all_estimated_criteria')
        summary['known_text_criteria_applicable'] = False
        return summary
    return {'recordings': len(primary), 'private_recordings': private_count,
            'original_positive': sum(r['detected'] for r in original_labels.values()),
            'primary_double_negative': sum(not primary_labels[k]['detected'] and not repeat_labels[k]['detected']
                                           for k in primary_labels),
            'selected_verified_negative': len(selected),
            'selected_presets': dict(Counter(c['selected_preset'] for c in cases.values()
                                            if c['status'] == 'verified_negative')),
            'primary_quality': descriptive_quality(primary), 'selected_quality': descriptive_quality(selected),
            'excluded_from_fresh_denominator': True, 'individual_private_fingerprints_excluded': True,
            'private_audio_or_transcripts_exported': False}


def private_regressions(directory):
    plan_path, api_path = directory/'filtered_regressions.plan.json', directory/'filtered_regressions.api.json'
    plan, api = load(plan_path), load(api_path)
    ids = {i['id'] for i in plan['items']}
    if (api['measurement_complete'] is not True or api['plan_sha256'] != digest(plan_path) or
            api['candidate'] != plan['candidate'] or
            api['render_report_sha256'] != digest(directory/'filtered_regressions.render.json') or
            len(api['http_attempts']) > plan['measurement']['maximum_http_attempts'] or
            sum(r.get('status') == 200 for r in api['http_attempts']) != len(api['clips']) or
            len(api['clips']) != 3*len(ids)+3):
        raise ValueError('Incomplete private regression measurements')
    original = indexed([r for r in api['clips'] if r['role'] == 'original'], ids, 'regression original')
    processed = indexed([r for r in api['clips'] if r['role'] == 'processed'], ids, 'regression processed')
    repeated = indexed([r for r in api['clips'] if r['role'] == 'processed_repeat'], ids, 'regression repeat')
    fresh_plan = load(directory/'filtered_eval.plan.json')
    sentinel = next(i for i in fresh_plan['items'] if i['id'] == api['sentinel_source_id'])
    controls = [r for r in api['clips'] if r['role'] == 'original_sentinel']
    if sorted(r['ordinal'] for r in controls) != [0, len(ids), 2*len(ids)]:
        raise ValueError('Private regression service-control schedule changed')
    for control in controls:
        if control['id'] != sentinel['id']:
            raise ValueError('Wrong private regression service control')
        checked(control, sentinel['sha256'])
    renders = indexed(load(directory/'filtered_regressions.render.json'), ids, 'regression render')
    metrics = {}
    for field, suffix in (('content', 'asr'), ('secondary_content', 'asr_turbo'), ('voice', 'voice_quality')):
        document = load(directory/('filtered_regressions.'+suffix+'.json'))
        if document['complete'] is not True or document['plan_sha256'] != digest(plan_path):
            raise ValueError('Incomplete or mismatched private quality measurements')
        metrics[field] = indexed(document['clips'], ids, 'regression '+field)
    rows = []
    for item in plan['items']:
        key = item['id']
        rendered = renders[key]
        if (rendered['source_sha256'] != item['sha256'] or rendered['format_and_length_preserved'] is not True or
                rendered['finite_unclipped'] is not True or rendered['inference']['settings'] != plan['settings'] or
                rendered['inference']['checkpoint_sha256'] != plan['candidate']['model_sha256'] or
                rendered['inference']['source_commit'] != plan['candidate']['source_commit'] or
                digest(ROOT/rendered['output']) != rendered['output_sha256']):
            raise ValueError('Private reconstruction differs from its frozen recipe')
        checked(original[key], item['sha256'])
        for observation in (processed[key], repeated[key]):
            checked(observation, renders[key]['output_sha256'])
        for group in metrics.values():
            if (group[key]['source_sha256'] != item['sha256'] or
                    group[key]['output_sha256'] != renders[key]['output_sha256']):
                raise ValueError('Private quality measured different files')
        rows.append({'id': key, 'source_sha256': item['sha256'], 'output_sha256': renders[key]['output_sha256'],
                     'content': content_metrics(metrics['content'][key]),
                     'secondary_content': content_metrics(metrics['secondary_content'][key]),
                     'voice': metrics['voice'][key]})
    workflow_path = directory/'filtered_private_workflow.api.json'
    workflow = load(workflow_path)
    protocol_path = directory/'filtered_private_workflow.plan.json'
    protocol = load(protocol_path)
    if (workflow['measurement_complete'] is not True or workflow['protocol_sha256'] != digest(protocol_path) or
            protocol['required_negative_checks'] != 2 or
            protocol['fallback_presets'] != ['full-flow', 'strong'] or protocol['code_sha256'] != plan['code_sha256'] or
            len(workflow['http_attempts']) > protocol['maximum_http_attempts']):
        raise ValueError('Incomplete private fallback workflow')
    cases = indexed(workflow['cases'], ids, 'private workflow')
    controls = workflow['service_controls']
    needed = sum(c['fallback_needed'] for c in cases.values())
    if sorted(c['label'] for c in controls) != (['after_fallback', 'before_fallback'] if needed else []):
        raise ValueError('Private fallback service controls are missing')
    if any(c['http_status'] != 200 or c['detected'] is not True for c in controls):
        raise ValueError('Private fallback service controls did not pass')
    for row in rows:
        case = cases[row['id']]
        if (case['source_sha256'] != row['source_sha256'] or
                case['fallback_needed'] != (processed[row['id']]['detected'] or repeated[row['id']]['detected'])):
            raise ValueError('Private fallback selection used different observations')
        if case['status'] == 'verified_negative' and case['negative_checks'] != 2:
            raise ValueError('Private selection lacks two negative checks')
        if case['status'] == 'verified_negative' and digest(ROOT/case['output']) != case['output_sha256']:
            raise ValueError('Private selected file changed after verification')
        if (not case['fallback_needed'] and case['status'] == 'verified_negative' and
                (case['selected_preset'] != 'filtered-prior' or case['output_sha256'] != row['output_sha256'])):
            raise ValueError('Private workflow silently replaced a primary negative file')
        if case['fallback_needed'] and case['status'] == 'verified_negative':
            audit_path = (ROOT/case['output']).with_suffix('.audit.json')
            audit = load(audit_path)
            verification_summary(audit, {})
            if (audit['output_sha256'] != case['output_sha256'] or audit['source_sha256'] != case['source_sha256'] or
                    audit['required_negative_checks'] != 2 or digest(audit_path) != case['audit_sha256'] or
                    audit['verification_cli_code_sha256'] != protocol['code_sha256']['tools/verify_reconstruction.py']):
                raise ValueError('Private fallback audit measured different files')
    fallback, _ = load_selected_fallback_quality(directory, 'filtered_private_fallback', cases, workflow_path)
    selected = selected_quality_rows(rows, cases, fallback)
    summary = regression_summary(rows, selected, original, processed, repeated, cases,
                                 sum(i['private'] for i in plan['items']))
    summary.update(http_attempts=len(api['http_attempts'])+len(workflow['http_attempts']),
                   service_controls=len(api['clips'])-3*len(ids)+len(workflow['service_controls']),
                   positive_service_controls=sum(r['detected'] for r in api['clips'] if r['role'] == 'original_sentinel')+
                                             sum(r['detected'] for r in workflow['service_controls']))
    return summary


def build_report(directory):
    fresh = build_vocoder_component(directory, 'filtered_v1', 'filtered_eval.plan.json')
    long = build_vocoder_component(directory, 'filtered_long', 'filtered_long_eval.plan.json')
    if fresh['candidate'] != long['candidate'] or fresh['code_sha256'] != long['code_sha256']:
        raise ValueError('Long supplement changed the frozen recipe')
    rows = fresh['clips']+long['clips']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sources across filtered-prior cohorts')
    generation_plan = load(directory/'filtered_generation.plan.json')
    native_ids = {i['id'] for i in generation_plan['items']}
    receipts = indexed([r for r in load(directory/'budget.json')['requests'] if r['id'] in native_ids],
                       native_ids, 'filtered-prior generation receipt')
    for item in load(directory/'filtered_eval.plan.json')['items']:
        if receipts[item['id']]['status'] != 'completed' or receipts[item['id']]['output_sha256'] != item['sha256']:
            raise ValueError('Fresh source lacks a matching generation receipt')
    smoke = load(directory/'filtered_verified_cli_smoke.audit.json')
    smoke_control = load(directory/'filtered_cli_smoke.complete.json')
    smoke_protocol = load(directory/'filtered_cli_smoke.protocol.json')
    if (smoke_control['status'] != 'passed' or smoke_control['positive_controls'] != 2 or
            smoke['required_negative_checks'] != 2 or
            smoke['verification_cli_code_sha256'] != smoke_protocol['code_sha256']['tools/verify_reconstruction.py'] or
            smoke_protocol['code_sha256'] != fresh['code_sha256'] or
            digest(Path(smoke['output'])) != smoke['output_sha256']):
        raise ValueError('Packaged two-confirmation CLI integration is incomplete')
    parity = load(directory/'filtered_prior_package_parity.json')
    if (parity['all_decoded_pcm_exact'] is not True or
            parity['code_sha256'] != fresh['code_sha256']['tools/reconstruct_filtered_prior.py'] or
            len(parity['cases']) != 2 or any(r['decoded_pcm_exact'] is not True for r in parity['cases'])):
        raise ValueError('Packaged development parity did not pass')
    workflow = public_workflow(directory, rows)
    regressions = private_regressions(directory)
    return {'schema_version': 1, 'candidate': fresh['candidate'], 'settings': fresh['settings'],
            'inference_source_commit': fresh['candidate']['project_source_commit'],
            'inference_code_sha256': fresh['code_sha256'], 'upstream_model_trained_by_this_project': False,
            'selection': {'fresh_recordings': len(fresh['clips']),
                          'fresh_voice_ids': len({i['voice'] for i in generation_plan['items']}),
                          'fresh_languages': sorted({i['language'] for i in generation_plan['items']}),
                          'english_note': generation_plan['english_note'],
                          'long_supplement': load(directory/'filtered_long_eval.plan.json')['selection'],
                          'no_independent_population_or_universal_claim': True},
            'summary': paired_counts(rows),
            'by_provenance': {key: paired_counts([r for r in rows if r['provenance_group'] == key])
                              for key in sorted({r['provenance_group'] for r in rows})},
            'fresh_study': {k: v for k, v in fresh.items() if k not in ('clips', 'metrics')},
            'long_study': {k: v for k, v in long.items() if k not in ('clips', 'metrics')},
            'quality_metrics': {**fresh['metrics'], **quality_summary(rows)},
            'primary_quality_failures': [{'id': r['id'], 'criteria': estimated_quality_gate(r)}
                                         for r in rows if not all(estimated_quality_gate(r).values())],
            'fixed_fallback_workflow': workflow, 'spent_regressions': regressions,
            'verification_cli_smoke': verification_summary(smoke, load(directory/'verification_environment.json')),
            'packaged_development_parity': parity,
            'native_generation': {'requests': len(receipts), 'characters': sum(r['characters'] for r in receipts.values()),
                                   'reused_long_sources': len(long['clips'])},
            'environment': {name: load(directory/(name+'_environment.json')) for name in ('asr', 'voice')},
            'all_primary_responses_negative': all(not r['processed_detected'] and not r['processed_repeat_detected'] for r in rows),
            'service_controls_all_positive': all(c['service_controls'] == c['positive_service_controls']
                                                for c in (fresh, long, workflow, regressions)),
            'interpretation': 'Whole-file measurements of a frozen recipe. Repeats, fallbacks, controls, reused sources, and spent regressions do not inflate fresh removal counts. Quality scores are estimates; no physical-erasure, listening-approval, or future-detector guarantee.',
            'clips': rows, 'study_completed_at': workflow['completed_at'],
            'exported_at': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.directory)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
