"""Export complete frozen-study observations, including every planned failure.

Raw audio, transcripts, embeddings, provider identifiers and local paths remain
local. All seven gates come from the prospective freeze, not observed results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export(plan_path, freeze_path, directory):
    plan, freeze = load(plan_path), load(freeze_path)
    if digest(plan_path) != freeze['generation_plan_sha256']:
        raise ValueError('Plan does not match prospective freeze')
    for name, checksum in freeze['source_files_sha256'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError('Frozen source/evaluation files changed')
    generation_path = directory/'generation.json'
    generation = load(generation_path)
    baseline, candidates = load(directory/'baseline.json'), load(directory/'candidates.json')
    if generation['generation_plan_sha256'] != digest(plan_path):
        raise ValueError('Generation receipt belongs to another prospective plan')
    for report in (generation, baseline, candidates):
        if not report['complete'] or report['freeze_sha256'] != digest(freeze_path):
            raise ValueError('All phases must be complete and bound to the same freeze')
    materialized = directory/'evaluation.plan.json'
    if load(materialized)['generation_receipt_sha256'] != digest(generation_path):
        raise ValueError('Materialized evaluation inputs differ from generation receipt')
    if any(r['generation_receipt_sha256'] != digest(generation_path) for r in (baseline, candidates)):
        raise ValueError('Endpoint phases refer to another generation receipt')
    if baseline['completed_at'] > candidates['started_at']:
        raise ValueError('Original baseline was not completed before candidate checks')
    scores = {m: load(directory/'scores'/('final_'+m+'.json')) for m in ('small', 'turbo', 'mms', 'voice')}
    originals = {m: load(directory/'scores'/('originals_'+m+'.json')) for m in ('small', 'turbo', 'mms')}
    ids = [i['id'] for i in plan['items']]
    for report in [generation, baseline, candidates, *scores.values(), *originals.values()]:
        rows = report['requests'] if report is generation else report['cases'] if report in (baseline, candidates) else report['clips']
        if [r['id'] for r in rows] != ids:
            raise ValueError('Every planned request must be retained, once and in plan order')
    for metric, report in scores.items():
        if not report['complete'] or report['originals_only'] or report['plan_sha256'] != digest(materialized):
            raise ValueError('Final quality report is incomplete or belongs to another plan')
    for metric, report in originals.items():
        if (not report['complete'] or not report['originals_only'] or report['candidate_outputs_read'] or
                report['plan_sha256'] != digest(materialized)):
            raise ValueError('Original-only calibration receipts are required')
        if report['completed_at'] > candidates['started_at']:
            raise ValueError('Original recognizer calibration did not precede candidate selection')
    gates = freeze['quality_criteria']
    rows = []
    for index, item in enumerate(plan['items']):
        source, original, selected = generation['requests'][index], baseline['cases'][index], candidates['cases'][index]
        if original['source_sha256'] != source['sha256'] or selected['source_sha256'] != source['sha256']:
            raise ValueError('Source identities differ between phases')
        source_path = ROOT/source['source']
        if digest(source_path) != source['sha256'] or digest(ROOT/source['raw_source']) != source['raw_sha256']:
            raise ValueError('Generated source or original provider receipt changed')
        original_checks = [c for c in baseline['checks'] if c['id'] == item['id']]
        if (len(original_checks) != 2 or [c['ordinal'] for c in original_checks] != [1, 2] or
                any(c['sha256'] != source['sha256'] or c['detected'] != original['detected'] for c in original_checks)):
            raise ValueError('Original repeated responses are not consistent exact-file checks')
        row = {'id': item['id'], 'language': item['language'], 'model': item['model'],
               'source_sha256': source['sha256'], 'raw_provider_sha256': source['raw_sha256'],
               'marked_provenance': item['marked_provenance'], 'original_detected': original['detected'],
               'source_format': {k: source[k] for k in ('sample_rate', 'samples', 'seconds', 'channels')},
               'status': selected['status'], 'original_calibration': {}, 'attempts': []}
        for metric, report in originals.items():
            score = report['clips'][index]
            if score['source_sha256'] != source['sha256']:
                raise ValueError('Original calibration source changed')
            row['original_calibration'][metric] = {k: score[k] for k in (
                'original_reference_error', 'original_reference_error_raw_script', 'original_calibration_eligible')}
        row['calibrated_checkpoint_count'] = sum(s['original_calibration_eligible'] for s in row['original_calibration'].values())
        row['source_calibration_sufficient'] = row['calibrated_checkpoint_count'] >= 2
        for attempt in selected['attempts']:
            model = attempt['model']
            result = {k: attempt[k] for k in ('preset', 'state', 'sha256', 'detected') if k in attempt}
            result['content_guard_passed'] = model['content_guard_passed']
            result['paired_disagreement'] = model['paired_disagreement']
            for key in ('rejection_reason', 'source_job_reused', 'inference_seconds'):
                if key in model:
                    result[key] = model[key]
            if attempt.get('sha256'):
                checks = [c for c in candidates['checks'] if c['id'] == item['id'] and c['sha256'] == attempt['sha256']]
                if len(checks) != 2 or any(c['detected'] != attempt['detected'] for c in checks):
                    raise ValueError('Candidate requires two consistent exact-file responses')
                result['detector_checks'] = [{k: c[k] for k in ('ordinal', 'http_status', 'detected', 'checked_at')} for c in checks]
            row['attempts'].append(result)
        row.update(quality_gates=None, all_seven_quality_gates_passed=False, qualified_checked_output=False,
                   positive_to_negative_transition=False)
        if selected['status'] == 'verified_negative':
            output = ROOT/item['verified_output']
            if digest(output) != selected['output_sha256'] or selected['negative_checks'] != 2:
                raise ValueError('Published output differs from twice-negative selected file')
            measured = {m: scores[m]['clips'][index] for m in scores}
            if any(s['output_sha256'] != selected['output_sha256'] or s['source_sha256'] != source['sha256'] or
                   s['status'] != 'scored' for s in measured.values()):
                raise ValueError('Quality metrics are not bound to the selected output')
            for metric in ('small', 'turbo', 'mms'):
                if measured[metric]['original_reference_error'] != originals[metric]['clips'][index]['original_reference_error']:
                    raise ValueError('Original reference scores changed after selection')
            quality = {
                'turbo_paired_disagreement': (measured['turbo']['paired_disagreement'], gates['turbo_paired_disagreement_maximum'], 'maximum'),
                'turbo_reference_error_delta': (measured['turbo']['reference_error_delta'], gates['turbo_reference_error_delta_maximum'], 'maximum'),
                'small_reference_error_delta': (measured['small']['reference_error_delta'], gates['small_reference_error_delta_maximum'], 'maximum'),
                'mms_reference_error_delta': (measured['mms']['reference_error_delta'], gates['mms_reference_error_delta_maximum'], 'maximum'),
                'mms_paired_disagreement': (measured['mms']['paired_disagreement'], gates['mms_paired_disagreement_maximum'], 'maximum'),
                'three_window_speaker_minimum': (measured['voice']['speaker_minimum'], gates['three_window_speaker_minimum'], 'minimum'),
                'dnsmos_overall_delta': (measured['voice']['dnsmos_delta']['overall'], gates['dnsmos_overall_delta_minimum'], 'minimum')}
            row['quality_gates'] = {name: {'value': value, 'threshold': threshold, 'direction': direction,
                'passed': value is not None and (value <= threshold+1e-12 if direction == 'maximum' else value >= threshold-1e-12)}
                for name, (value, threshold, direction) in quality.items()}
            passed = all(g['passed'] for g in row['quality_gates'].values())
            row.update(output_sha256=selected['output_sha256'], selected_preset=selected['selected_preset'], negative_checks=2,
                       all_seven_quality_gates_passed=passed, qualified_checked_output=passed and row['source_calibration_sufficient'],
                       positive_to_negative_transition=original['detected'],
                       quality_estimates={m: {k: v for k, v in s.items() if k not in ('id', 'source_sha256', 'output_sha256', 'language')}
                                          for m, s in measured.items()})
        rows.append(row)
    controls = [c for c in candidates['checks'] if c['role'] in ('opening_positive_control', 'closing_positive_control')]
    if len(controls) != 4 or any(c['detected'] is not True for c in controls):
        raise ValueError('Opening and closing positive controls are required')
    positive_sources = {r['source_sha256'] for r in rows if r['original_detected'] and r['marked_provenance']}
    if any(c['sha256'] not in positive_sources or c['http_status'] != 200 for c in controls):
        raise ValueError('Endpoint controls must refer to measured-positive marked originals')
    return {'status': 'Complete observations; limited generated cohort, not universal removal',
            'generation_plan_sha256': digest(plan_path), 'freeze_sha256': digest(freeze_path),
            'candidate_implementation_commit': freeze['candidate_implementation_commit'],
            'evaluation_implementation_commit': freeze['evaluation_implementation_commit'],
            'planned_requests': len(rows), 'distinct_generated_recordings': len({r['source_sha256'] for r in rows}),
            'matched_nominal_controls': sum(not r['marked_provenance'] for r in rows),
            'marked_provenance_requests': sum(r['marked_provenance'] for r in rows),
            'original_measured_positive': sum(r['original_detected'] for r in rows),
            'selected_twice_negative': sum(r['status'] == 'verified_negative' for r in rows),
            'all_seven_quality_gates_passed': sum(r['all_seven_quality_gates_passed'] for r in rows),
            'qualified_checked_outputs': sum(r['qualified_checked_output'] for r in rows),
            'qualified_marked_positive_to_negative': sum(r['marked_provenance'] and r['original_detected'] and r['qualified_checked_output'] for r in rows),
            'human_listening_approval': False, 'physical_erasure_proven': False, 'clips': rows,
            'endpoint_controls': controls,
            'http_attempts': {name: len(r['http_attempts']) for name, r in [('baseline', baseline), ('candidates', candidates)]},
            'metrics': {name: r['metric'] for name, r in scores.items()},
            'limitations': plan['scope_limits'], 'candidate_phase_completed_at': candidates['completed_at'],
            'exported_at': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'audio/codec_reliability_plan.json')
    parser.add_argument('--freeze', type=Path, default=ROOT/'results/codec_reliability.freeze.json')
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability/codec_fresh')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_reliability.json')
    args = parser.parse_args()
    result = export(args.plan, args.freeze, args.directory)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
