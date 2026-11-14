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
             'mimi_primary_repair_endpoint_queue.json'),
            ('ta2_roundtrip', 'ta2_roundtrip_quality', 'ta2_', ('tempo', 'pitch'),
             'ta2_roundtrip_endpoint_queue.json'),
            ('mimi_pcm24_repair', 'mimi_pcm24_repair_quality_v2', 'ta2_pcm_', ('native',),
             'mimi_pcm24_repair_endpoint_queue_v2.json')):
        study = load(directory/study_name/'result.json')
        quality = load(directory/quality_name/'result.json')
        load(directory/queue)
        if len(study['candidates']) != len(variants):
            raise ValueError('Every planned treatment must remain in the report')
        if study_name == 'mimi_pcm24_repair':
            rejected = study['unrendered_candidates']
            if (study['donor_order_predeclared'] != ['native', 'prior'] or len(rejected) != 1 or
                    rejected[0]['label'] != 'prior' or rejected[0]['status'] != 'no_safe_donor_region'):
                raise ValueError('The unrendered prior donor must remain an unsafe-selection rejection')
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
            elif study_name == 'mimi_primary_repair':
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
            elif study_name == 'ta2_roundtrip':
                lineage = {'type': 'Whole-file Rubber Band forward/inverse reconstruction',
                           'filters': scored['filters'], 'ffmpeg_version': study['ffmpeg_version'],
                           'ffmpeg_executable_sha256': study['ffmpeg_executable_sha256'],
                           'conditioning_reflection_seconds_each_side': study['conditioning_reflection_seconds_each_side'],
                           'conditioning_padding_removed_before_publication': True,
                           'global_alignment_samples': scored['global_alignment_samples'],
                           'no_local_alignment': True, 'render_seconds': scored['render_seconds'],
                           'aligned_waveform_correlation': scored['aligned_waveform_correlation']}
            else:
                commit = study['pcm24_blend_implementation_commit']
                blob = subprocess.check_output(['git', 'show', commit+':experiments/pcm24_repair.py'], cwd=ROOT)
                if hashlib.sha256(blob).hexdigest() != study['pcm24_blend_code_sha256']:
                    raise ValueError('PCM24 blending differs from its committed source')
                if (not scored['zero_operation_exact'] or not scored['encoded_untouched_samples_preserved'] or
                        scored['zero_operation_sha256'] != study['initial_candidate_sha256'] or
                        scored['repair']['gain'] != 1.):
                    raise ValueError('The numerical repair did not preserve its stated sample invariants')
                lineage = {'type': 'Primary-guided donor repair preserving untouched PCM24 samples',
                           'initial_candidate_sha256': study['initial_candidate_sha256'],
                           'generated_donor_sha256': scored['donor_sha256'],
                           'donor_order_predeclared': study['donor_order_predeclared'],
                           'selection': scored['repair_selection'], 'repair': scored['repair'],
                           'blend_implementation_commit': commit, 'blend_source_sha256': study['pcm24_blend_code_sha256'],
                           'zero_operation_sha256': scored['zero_operation_sha256'],
                           'zero_operation_exact': True, 'encoded_untouched_samples_preserved': True}
            rows.append(candidate(directory, case, label, directory/study_name/str(variant), scored,
                                  study['original_reference_error'], quality, label, 'ta', lineage))
    if len(rows) != 10 or len({r['source_sha256'] for r in rows}) != 1:
        raise ValueError('Keep all ten rendered treatments of the one spent source')
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
    review = load(directory/'pcm24_repair_numerical_review.json')
    numerical_review = {k: review[k] for k in ('source_candidate_sha256', 'patched_sha256',
        'outside_region_samples', 'outside_region_changed_samples', 'maximum_outside_difference_pcm24_steps',
        'base_peak', 'nominal_headroom', 'applied_gain', 'new_asr_calls', 'reference_text_used')}
    return {'kind': 'Spent second-Tamil split-band, Mimi, and primary-repair probes',
            'independent_source_recordings': 1, 'fresh_independent_sources': 0,
            'new_tts_requests': 0, 'frozen_primary_study_result_changed': False,
            'planned_treatments': 12, 'rendered_candidates': 10,
            'candidates': rows, 'unrendered_coverage_rejection': joint,
            'unrendered_donor_rejection': load(directory/'mimi_pcm24_repair/result.json')['unrendered_candidates'],
            'donor_rejection_receipt': load(directory/'mimi_pcm24_repair/selection_rejection_receipt.json'),
            'old_blending_numerical_review': numerical_review,
            'unsupported_character_consensus_review': consensus_review,
            'qualified_detector_negative_sources': int(any(r['qualified_detector_negative'] for r in rows)),
            'human_listening_approval': False, 'physical_erasure_proven': False,
            'limits': ['All twelve planned treatments reuse the same previously tested Sonic Preview recording.',
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
