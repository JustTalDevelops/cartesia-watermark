"""Curate every planned split-band/Mimi treatment of the spent Tamil source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.codec_followup_report import candidate
from experiments.vocoder_codec_tradeoff_report import load


def export(directory):
    case = 'codec_fresh_12_ta2_sonic-preview'
    rows = []
    for study_name, quality_name, prefix, variants, queue in (
            ('ta2_split_outside', 'ta2_split_outside_quality', 'ta2_', ('high_native', 'low_native', 'f0_native'),
             'ta2_split_outside_endpoint_queue.json'),
            ('mimi_ta2', 'mimi_ta2_quality', 'ta2_mimi', (16, 32), 'mimi_ta2_endpoint_queue.json'),
            ('mimi_primary_repair', 'mimi_primary_repair_quality', 'ta2_mimi_repair', (16, 32),
             'mimi_primary_repair_endpoint_queue.json')):
        study = load(directory/study_name/'result.json')
        quality = load(directory/quality_name/'result.json')
        load(directory/queue)
        if len(study['candidates']) != len(variants):
            raise ValueError('Every planned treatment must remain in the report')
        for variant in variants:
            label = prefix+str(variant)
            scored = next(c for c in study['candidates'] if c.get('label', c.get('num_quantizers')) == variant)
            if study_name == 'ta2_split_outside':
                lineage = {'type': 'Partial original-band reconstruction with one generated outside region',
                           'outside_native_filter': scored['outside_native_filter'],
                           'original_band_baseline_sha256': study['original_band_baseline_sha256'],
                           'generated_native6_sha256': study['native6_sha256'],
                           'original_waveform_retained': True, 'full_regeneration': False}
            elif study_name == 'mimi_ta2':
                lineage = {'type': 'Direct complete Mimi reconstruction', 'model': study['codec'],
                           'encoder_prefix_reused': study['encoder_prefix_reused'],
                           'num_quantizers': variant, 'nominal_bitrate_kbps': scored['nominal_bitrate_kbps'],
                           'reconstruction_seconds': scored['reconstruction_seconds'],
                           'guard_seconds': scored['guard_seconds'],
                           'timing_limit': 'Separate exploratory runs; excludes startup and is not a controlled bitrate timing comparison'}
            else:
                commit = subprocess.check_output(['git', 'rev-parse', '2e19c21'], cwd=ROOT, text=True).strip()
                blob = subprocess.check_output(['git', 'show', commit+':experiments/codec_primary_repair.py'], cwd=ROOT)
                if hashlib.sha256(blob).hexdigest() != study['selection_code_sha256']:
                    raise ValueError('Primary repair selection differs from its recorded source')
                lineage = {'type': 'One primary-guard-guided generated-donor repair of Mimi',
                           'initial_candidate_sha256': scored['initial_candidate_sha256'],
                           'native_donor_sha256': study['native_donor_sha256'], 'repair': scored['repair'],
                           'selection_code_commit': commit, 'selection_code_sha256': study['selection_code_sha256'],
                           'maximum_coverage': study['maximum_coverage'],
                           'processing_selection_uses_reference_text': False,
                           'processing_selection_uses_evaluation_asr': False}
            rows.append(candidate(directory, case, label, directory/study_name/str(variant), scored,
                                  study['original_reference_error'], quality, label, 'ta', lineage))
    if len(rows) != 7 or len({r['source_sha256'] for r in rows}) != 1:
        raise ValueError('Keep all seven treatments of the one spent source')
    joint = load(directory/'mimi_joint_repair/result.json')
    consensus_review = load(directory/'tamil_mimi_consensus_review.json')
    initial = next(r for r in rows if r['id'] == 'ta2_mimi16')
    if (joint['source_sha256'] != initial['source_sha256'] or
            joint['initial_candidate_sha256'] != initial['candidate_sha256'] or
            joint['maximum_coverage'] != .2 or joint['status'] != 'coverage_rejected' or
            joint['rendered_output'] or joint['new_http_requests'] != 0 or joint['new_asr_calls'] != 0):
        raise ValueError('Retain the unrendered joint proposal as a coverage rejection')
    if (consensus_review['candidate_sha256'] != next(r['candidate_sha256'] for r in rows if r['id'] == 'ta2_mimi32') or
            consensus_review['context_units_each_side'] != 16 or consensus_review['supported_isolated_substitutions'] or
            not consensus_review['raw_guard_rejection_retained'] or consensus_review['new_profile_applied']):
        raise ValueError('Unsupported consensus cannot override the raw speech rejection')
    return {'kind': 'Spent second-Tamil split-band, Mimi, and primary-repair probes',
            'independent_source_recordings': 1, 'fresh_independent_sources': 0,
            'new_tts_requests': 0, 'frozen_primary_study_result_changed': False,
            'planned_treatments': 8, 'rendered_candidates': 7,
            'candidates': rows, 'unrendered_coverage_rejection': joint,
            'unsupported_character_consensus_review': consensus_review,
            'qualified_detector_negative_sources': int(any(r['qualified_detector_negative'] for r in rows)),
            'human_listening_approval': False, 'physical_erasure_proven': False,
            'limits': ['All seven treatments reuse the same previously tested Sonic Preview recording.',
                       'Tamil processing and independent reference-delta thresholds remain one percent, with vowel marks preserved.',
                       'Split-band outputs retain original audio and cannot establish physical erasure.',
                       'The repair uses primary-ASR differences and unique independent timing context, generated donors only, and a twenty-percent coverage ceiling; final guards are rerun without transcript exceptions.',
                       'Reference text and the evaluation recognizer enter scoring only after processing is fixed.',
                       'Failed speech/quality checks remain failures and are not sent to the detector.',
                       'Repeated whole-file negatives with positive source brackets would still add no held-out source.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_ta2_followup.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.directory), indent=2, allow_nan=False)+'\n')
    print(args.output)
