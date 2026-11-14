"""Publish complete spent vocoder/Mimi trials, retaining quality rejections."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    result = json.loads(path.read_text())
    if result.get('complete') is not True:
        raise ValueError(f'Incomplete measurement: {path.name}')
    return result


def verify_endpoint(folder, assessment_path, source_sha, candidate_sha):
    endpoint = load(folder / 'result.json')
    detected = endpoint['candidate_detected']
    if type(detected) is not bool:
        raise ValueError('A repeated candidate decision is required')
    expected = [('opening_original', True)] * 2 + [('candidate', detected)] * 2 + [('closing_original', True)] * 2
    if (endpoint['source_sha256'] != source_sha or endpoint['candidate_sha256'] != candidate_sha or
            endpoint['assessment_sha256'] != digest(assessment_path) or
            [(c['role'], c['detected']) for c in endpoint['checks']] != expected or
            len(endpoint['http_attempts']) != 6 or
            any(c['http_status'] != 200 for c in endpoint['checks'])):
        raise ValueError('Detector evidence does not bind the exact complete candidate and controls')
    for records in (endpoint['checks'], endpoint['http_attempts']):
        if len(records) != 6:
            raise ValueError('Exactly six complete-file requests are required')
        for index, record in enumerate(records):
            role = expected[index][0]
            checksum = candidate_sha if role == 'candidate' else source_sha
            if (record['role'] != role or record['ordinal'] != index % 2 + 1 or
                    record['sha256'] != checksum or record['http_status'] != 200):
                raise ValueError('A request or response is bound to different audio or controls')
    starts = [datetime.fromisoformat(c['started_at']) for c in endpoint['http_attempts']]
    if any((b - a).total_seconds() < 79.9 for a, b in zip(starts, starts[1:])):
        raise ValueError('Detector pacing changed')
    return endpoint


def export(directory):
    quality_main = load(directory / 'vocoder_mimi_quality/result.json')
    quality_ta2 = load(directory / 'ta2_vocoder_quality/result.json')
    quality_mid = load(directory / 'ta1_mid_vocoder_quality/result.json')
    mimi_review = load(directory / 'mimi8_arabic_strong_review/result.json')
    specifications = [
        ('ta_gentle', 'codec_tamil_full_vocoder', '0.5', quality_main),
        ('ta_smoothed', 'codec_tamil_full_vocoder', '1.0', quality_main),
        ('ta_mid', 'codec_ta1_mid_vocoder', '0.75', quality_mid),
        ('ta2_gentle', 'codec_ta2_full_vocoder', '0.5', quality_ta2),
        ('ta2_smoothed', 'codec_ta2_full_vocoder', '1.0', quality_ta2),
        ('ar_mimi16', 'mimi_arabic_development', '16', quality_main),
    ]
    rows = []
    for label, study, variant, quality in specifications:
        folder = directory / study / variant
        study_report = load(directory / study / 'result.json')
        candidate = folder / 'candidate.flac'
        assessment_path = folder / 'assessment.json'
        assessed = load(assessment_path)
        scored = next(c for c in study_report['candidates'] if c['output_sha256'] == digest(candidate))
        evaluation = next(c for c in quality['candidates'] if c['id'] == label)
        voice = next(c for c in quality['voice']['clips'] if c['id'] == label)
        source_sha, candidate_sha = assessed['source_sha256'], digest(candidate)
        if any(r['source_sha256'] != source_sha or r['output_sha256'] != candidate_sha
               for r in (scored, assessed, evaluation, voice)):
            raise ValueError('A quality measurement refers to different audio')
        original = quality['evaluation_checkpoints'][evaluation['evaluation_checkpoint']]
        if original['source_sha256'] != source_sha or not quality['voice']['complete']:
            raise ValueError('Original calibration or voice measurement does not match the source')
        threshold = assessed['maximum_paired_disagreement']
        primary, secondary = assessed['primary_guard'], assessed['secondary_guard']
        gates = {
            'evaluation_checkpoint_reference_delta': evaluation['reference_error_delta'] <= threshold + 1e-12,
            'primary_reference_delta': scored['reference_error_delta_evaluation_only'][primary] <= threshold + 1e-12,
            'secondary_reference_delta': scored['reference_error_delta_evaluation_only'][secondary] <= threshold + 1e-12,
            'primary_paired': assessed['paired_disagreement'][primary] <= threshold + 1e-12,
            'secondary_paired': assessed['paired_disagreement'][secondary] <= threshold + 1e-12,
            'speaker_minimum': voice['speaker_minimum'] >= .95,
            'dnsmos_overall': voice['dnsmos_delta']['overall'] >= -.2,
        }
        calibration_ceiling = .1 if primary == 'large' and secondary == 'mms' else .15
        calibrated_checkpoints = sum(e <= calibration_ceiling for e in study_report['original_reference_error'].values())
        calibrated_checkpoints += original['original_reference_error'] <= calibration_ceiling
        calibrated = calibrated_checkpoints >= 2
        endpoint = None
        if (folder / 'endpoint').exists():
            endpoint = verify_endpoint(folder / 'endpoint', assessment_path, source_sha, candidate_sha)
        elif assessed['content_guard_passed']:
            raise ValueError('A planned speech-accepted candidate has no detector result')
        qualified = calibrated and assessed['content_guard_passed'] and all(gates.values())
        cleared = qualified and endpoint is not None and endpoint['candidate_detected'] is False
        rows.append({'id': label, 'source_sha256': source_sha, 'candidate_sha256': candidate_sha,
                     'settings': {k: scored[k] for k in ('sigma_frequency', 'sigma_time', 'num_quantizers',
                                  'nominal_bitrate_kbps') if k in scored},
                     'assessment': assessed, 'reference_errors': {
                         'original_guards': study_report['original_reference_error'],
                         'processed_guards': scored['reference_error_evaluation_only'],
                         'guard_deltas': scored['reference_error_delta_evaluation_only'],
                         'evaluation_only': evaluation, 'evaluation_checkpoint': original},
                     'voice': voice, 'quality_gate_results': gates,
                     'calibrated_original_checkpoints': calibrated_checkpoints,
                     'original_calibration_ceiling': calibration_ceiling,
                     'all_seven_estimated_quality_gates_passed': all(gates.values()),
                     'quality_qualified': qualified, 'detector': endpoint,
                     'qualified_detector_negative': cleared,
                     'status': ('qualified_detector_negative' if cleared else
                                'still_detected_twice' if endpoint and endpoint['candidate_detected'] else
                                'detector_negative_but_quality_rejected' if endpoint else
                                'speech_rejected_not_detector_tested')})
    # Keep the unqueried, lower-bitrate speech failure and the later stronger
    # checkpoint reassessment. Neither substitutes for the original rejection.
    mimi = load(directory / 'mimi_arabic_development/result.json')
    lower = next(c for c in mimi['candidates'] if c['num_quantizers'] == 8)
    if (digest(directory / 'mimi_arabic_development/8/candidate.flac') != lower['output_sha256'] or
            lower['content_guard_passed'] or mimi_review['strong_guard_assessment']['content_guard_passed'] or
            lower['output_sha256'] != mimi_review['output_sha256'] or
            lower['source_sha256'] != mimi_review['source_sha256']):
        raise ValueError('Retain both actual lower-bitrate speech rejections')
    rows.append({'id': 'ar_mimi8', 'source_sha256': lower['source_sha256'],
                 'candidate_sha256': lower['output_sha256'], 'assessment': lower,
                 'stronger_checkpoint_reassessment': mimi_review, 'detector': None,
                 'qualified_detector_negative': False,
                 'status': 'speech_rejected_not_detector_tested'})
    source_files = list((directory / 'codec_fresh/originals').glob('*.flac'))
    for source_sha in {r['source_sha256'] for r in rows}:
        if not any(digest(p) == source_sha for p in source_files):
            raise ValueError('A complete original source is missing or changed')
    return {'kind': 'Spent vocoder smoothing and alternative-codec tradeoffs',
            'independent_source_recordings': 3, 'fresh_independent_sources': 0,
            'new_tts_requests': 0, 'frozen_primary_study_result_changed': False,
            'candidates': rows, 'qualified_detector_negatives': sum(r['qualified_detector_negative'] for r in rows),
            'upstream_mimi_assets': mimi['codec'], 'human_listening_approval': False,
            'physical_erasure_proven': False,
            'limits': ['All three sources were used previously; no new held-out evidence.',
                       'Tamil uses the prospective one-percent character policy preserving vowel marks; Arabic uses three-percent words.',
                       'These checkpoint and normalization profiles differ from the frozen primary study.',
                       'A detector-negative file that damages speech remains a failed removal.',
                       'Whisper checkpoints share ancestry and do not constitute independent recognizer architectures.',
                       'Original calibration and voice scores are estimates, not human approval.',
                       'The frozen primary result remains six qualified marked sources out of fourteen.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / '.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/vocoder_codec_tradeoffs.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.directory), indent=2, allow_nan=False) + '\n')
    print(args.output)
