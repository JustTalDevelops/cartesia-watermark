"""Curate fixed band-focused vocoder trials on a spent Tamil counterexample."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.vocoder_codec_tradeoff_report import digest, load, verify_endpoint

PROTOTYPE_COMMIT = 'c6cae6b'


def export(directory):
    folder = directory / 'codec_ta1_band_smoothing'
    generation = load(folder / 'generation.json')
    guards = load(folder / 'assessment_summary.json')
    reference = load(folder / 'reference_evaluation.json')
    quality = load(directory / 'ta1_band_smoothing_quality/result.json')
    queue = load(directory / 'band_smoothing_endpoint_queue.json')
    prototype = subprocess.check_output(['git', 'show', PROTOTYPE_COMMIT + ':experiments/probe_vocoder_smoothing.py'], cwd=ROOT)
    if hashlib.sha256(prototype).hexdigest() != generation['probe_sha256']:
        raise ValueError('Generation does not bind the published prototype')
    source_sha = generation['source_sha256']
    source = directory / 'codec_fresh/originals/codec_fresh_10_ta1_sonic-3.6.flac'
    if (digest(source) != source_sha or guards['source_sha256'] != source_sha or
            reference['source_sha256'] != source_sha or
            reference['assessment_summary_sha256'] != digest(folder / 'assessment_summary.json')):
        raise ValueError('Original, assessment and evaluation provenance differ')
    rows = []
    for label, quality_id in (('f1_t1', 'ta_band1'), ('f1.5_t1.5', 'ta_band15')):
        candidate = folder / label / 'candidate.flac'
        assessed_path = candidate.with_name('assessment.json')
        assessed = load(assessed_path)
        checksum = digest(candidate)
        generated = next(c for c in generation['candidates'] if c['output_sha256'] == checksum)
        scored = next(c for c in reference['candidates'] if c['label'] == label)
        evaluated = next(c for c in quality['candidates'] if c['id'] == quality_id)
        voice = next(c for c in quality['voice']['clips'] if c['id'] == quality_id)
        if any(c['source_sha256'] != source_sha or c['output_sha256'] != checksum
               for c in (generated, assessed, scored, evaluated, voice)):
            raise ValueError('An estimate refers to different audio')
        gates = {'large_paired': assessed['paired_disagreement']['large'] <= .01 + 1e-12,
                 'mms_paired': assessed['paired_disagreement']['mms'] <= .01 + 1e-12,
                 'large_reference_delta': scored['reference_error_delta_evaluation_only']['large'] <= .01 + 1e-12,
                 'mms_reference_delta': scored['reference_error_delta_evaluation_only']['mms'] <= .01 + 1e-12,
                 'small_reference_delta': evaluated['reference_error_delta'] <= .01 + 1e-12,
                 'speaker_minimum': voice['speaker_minimum'] >= .95,
                 'dnsmos_overall': voice['dnsmos_delta']['overall'] >= -.2}
        original_small = quality['evaluation_checkpoints']['small']
        if original_small['source_sha256'] != source_sha:
            raise ValueError('Independent calibration refers to a different original')
        original_errors = dict(reference['original_reference_error'], small=original_small['original_reference_error'])
        calibrated = sum(error <= .1 for error in original_errors.values())
        qualifies = assessed['content_guard_passed'] and all(gates.values()) and calibrated >= 2
        queued = next(c for c in queue['checks'] if c['label'] == label)
        if queued['candidate_sha256'] != checksum or queued['quality_gate_results'] != gates:
            raise ValueError('The detector eligibility receipt differs from these estimates')
        endpoint = None
        if (candidate.parent / 'endpoint').exists():
            endpoint = verify_endpoint(candidate.parent / 'endpoint', assessed_path, source_sha, checksum)
        elif qualifies and queued['status'] != 'not_reached_after_verified_output':
            raise ValueError('A planned eligible file is missing its detector protocol')
        cleared = qualifies and endpoint is not None and endpoint['candidate_detected'] is False
        rows.append({'id': label, 'source_sha256': source_sha, 'candidate_sha256': checksum,
                     'generation': generated, 'content_assessment': assessed,
                     'original_reference_error': original_errors,
                     'guard_reference_error': scored['reference_error_evaluation_only'],
                     'guard_reference_error_delta': scored['reference_error_delta_evaluation_only'],
                     'independent_small_evaluation': evaluated, 'voice': voice,
                     'calibrated_original_checkpoints': calibrated,
                     'quality_gate_results': gates, 'all_seven_estimated_quality_gates_passed': all(gates.values()),
                     'detector': endpoint, 'qualified_detector_negative': cleared,
                     'status': 'qualified_detector_negative' if cleared else queued['status']})
    return {'kind': 'Spent band-focused vocoder development', 'independent_source_recordings': 1,
            'fresh_independent_sources': 0, 'new_tts_requests': 0,
            'frozen_primary_study_result_changed': False, 'source_sha256': source_sha,
            'prototype_commit': subprocess.check_output(['git', 'rev-parse', PROTOTYPE_COMMIT], cwd=ROOT, text=True).strip(),
            'prototype_sha256': generation['probe_sha256'], 'settings': generation['settings'],
            'upstream_assets': generation['upstream_assets'],
            'recognizer_implementation_commit': guards['recognizer_implementation_commit'],
            'candidates': rows, 'qualified_detector_negative_sources': int(any(c['qualified_detector_negative'] for c in rows)),
            'human_listening_approval': False, 'physical_erasure_proven': False,
            'limits': ['One already-used source; different settings and repeated queries add no independent sources.',
                       'Tamil uses the prospective one-percent character policy preserving vowel marks, not the frozen word policy.',
                       'The stronger variant failed estimated naturalness and was not detector-tested.',
                       'A detector-negative result is not proof of physical erasure or reliability on new recordings.',
                       'The frozen primary result remains six qualified marked sources out of fourteen.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / '.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/vocoder_band_development.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.directory), indent=2, allow_nan=False) + '\n')
    print(args.output)
