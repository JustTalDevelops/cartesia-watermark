"""Curate spent follow-ups, keeping every measured quality/detector failure."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.vocoder_codec_tradeoff_report import digest, load, verify_endpoint


def original_evaluation(quality, case, metric):
    if 'original_calibration' in quality:
        return next(c for c in quality['original_calibration'] if c['id'] == case and c['metric'] == metric)
    return quality['evaluation_checkpoints'][metric]


def candidate(directory, case, label, folder, scored, original_guards, quality, evaluation_id, language, lineage,
              assessment_filename='assessment.json'):
    path = folder/'candidate.flac'
    assessment_path = folder/assessment_filename
    assessment = load(assessment_path)
    checksum = digest(path)
    source = directory/'codec_fresh/originals'/(case+'.flac')
    source_sha = digest(source)
    metric = 'large' if language == 'ar' else 'small'
    evaluation = next(c for c in quality['candidates'] if c['id'] == evaluation_id and c['evaluation_checkpoint'] == metric)
    original = original_evaluation(quality, case, metric)
    voice = next(c for c in quality['voice']['clips'] if c['id'] == evaluation_id)
    if any(c['source_sha256'] != source_sha or c['output_sha256'] != checksum
           for c in (assessment, scored, evaluation, voice)) or original['source_sha256'] != source_sha:
        raise ValueError('Source, candidate and quality estimates do not refer to the same complete files')
    primary, secondary = assessment['primary_guard'], assessment['secondary_guard']
    maximum = assessment['maximum_paired_disagreement']
    deltas = scored['reference_error_delta_evaluation_only']
    gates = {'primary_paired': assessment['paired_disagreement'][primary] <= maximum+1e-12,
             'secondary_paired': assessment['paired_disagreement'][secondary] <= maximum+1e-12,
             'primary_reference_delta': deltas[primary] <= maximum+1e-12,
             'secondary_reference_delta': deltas[secondary] <= maximum+1e-12,
             'evaluation_reference_delta': evaluation['reference_error_delta'] <= maximum+1e-12,
             'speaker_minimum': voice['speaker_minimum'] >= .95,
             'dnsmos_overall': voice['dnsmos_delta']['overall'] >= -.2}
    original_errors = dict(original_guards, **{metric: original['original_reference_error']})
    extra = None
    if language == 'ar':
        extra = next(c for c in quality['candidates'] if c['id'] == evaluation_id and c['evaluation_checkpoint'] == 'mms')
        extra_original = original_evaluation(quality, case, 'mms')
        if extra['source_sha256'] != source_sha or extra['output_sha256'] != checksum or extra_original['source_sha256'] != source_sha:
            raise ValueError('The additional Arabic recognizer evaluated different audio')
        original_errors['evaluation_mms'] = extra_original['original_reference_error']
        gates['additional_mms_reference_delta'] = extra['reference_error_delta'] <= .03+1e-12
    calibrated = sum(e <= (.1 if language == 'ta' else .15) for e in original_errors.values())
    qualified = assessment['content_guard_passed'] and all(gates.values()) and calibrated >= 2
    endpoint = None
    if (folder/'endpoint').exists():
        endpoint = verify_endpoint(folder/'endpoint', assessment_path, source_sha, checksum)
    elif qualified:
        raise ValueError('A planned quality-qualified candidate has no completed detector protocol')
    cleared = qualified and endpoint is not None and endpoint['candidate_detected'] is False
    return {'id': label, 'source_case': case, 'source_sha256': source_sha, 'candidate_sha256': checksum,
            'language': language, 'lineage': lineage, 'content_assessment': assessment,
            'original_reference_error': original_errors,
            'guard_reference_error': scored['reference_error_evaluation_only'],
            'guard_reference_error_delta': deltas, 'evaluation': evaluation, 'additional_mms_evaluation': extra,
            'voice': voice, 'quality_gate_results': gates, 'calibrated_original_checkpoints': calibrated,
            'quality_qualified': qualified, 'detector': endpoint, 'qualified_detector_negative': cleared,
            'status': 'qualified_detector_negative' if cleared else 'still_detected_twice' if endpoint else
                      'speech_rejected_not_detector_tested' if not assessment['content_guard_passed'] else
                      'quality_rejected_not_detector_tested'}


def packaged(directory, name, selected, endpoint_path):
    path = directory/name/'result.json'
    if not path.exists():
        return None
    report = load(path)
    if (report['source_sha256'] != selected['source_sha256'] or
            report['canonical_output_sha256'] != selected['candidate_sha256'] or
            not report['actual_asr_used'] or not report['exact_prototype_file_reproduction'] or
            report['fresh_independent_sources'] != 0 or report['new_http_requests'] != 0 or
            report['reference_text_used'] is not False or
            digest(directory/name/'candidate.flac') != selected['candidate_sha256'] or
            digest(directory/name/'metadata.json') != report['metadata_sha256'] or
            digest(endpoint_path) != report['original_research_endpoint_receipt_sha256']):
        raise ValueError('Packaged integration does not bind the exact checked output')
    for filename, checksum in report['source_files_sha256'].items():
        content = subprocess.check_output(['git', 'show', report['implementation_commit']+':'+filename], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError('The packaged implementation differs from its recorded Git blobs')
    return report


def export(directory):
    base = load(directory/'codec_band_remaining/result.json')
    quality = load(directory/'band_remaining_quality/result.json')
    load(directory/'band_remaining_endpoint_queue.json')
    rows = []
    for scored in base['candidates']:
        case = scored['id']
        original = next(c for c in base['original_calibration'] if c['id'] == case)
        rows.append(candidate(directory, case, 'band_prior_'+original['language'], directory/'codec_band_remaining'/case,
                              scored, original['original_reference_error'], quality, case, original['language'],
                              {'type': 'band-focused BigVGAN', 'sigmas': [1., 1.],
                               'band_hz': [1000, 1500, 3500, 4000], 'outside_sigmas': [.5, .5]}))
    case = 'codec_fresh_09_ru_sonic-preview'
    repair = load(directory/'ru_band_repair/result.json')
    repair_quality = load(directory/'ru_band_repair_quality/result.json')
    load(directory/'ru_band_repair_endpoint_queue.json')
    rows.append(candidate(directory, case, 'ru_band_prior_repair', directory/'ru_band_repair', repair['candidates'][0],
                          repair['original_reference_error'], repair_quality, 'ru_band_repaired', 'ru',
                          {'type': 'band prior plus bounded generated native donor',
                           'initial_candidate_sha256': repair['initial_candidate_sha256'],
                           'native_donor_sha256': repair['native_sha256'], 'repair': repair['repair'],
                           'historical_native_reproduction_exact': repair['exact_historical_native_reproduction']}))
    inverse = load(directory/'ko_inverse_band_mix/result.json')
    inverse_quality = load(directory/'ko_inverse_band_mix_quality/result.json')
    load(directory/'ko_inverse_mix_endpoint_queue.json')
    case = 'codec_fresh_07_ko_sonic-preview'
    for scored in inverse['candidates']:
        label = scored['label']
        rows.append(candidate(directory, case, 'ko_inverse_'+label, directory/'ko_inverse_band_mix'/label,
                              scored, inverse['original_reference_error'], inverse_quality, 'ko_inverse_'+label, 'ko',
                              {'type': 'generated native6 outside, generated band prior inside',
                               'outside_sha256': inverse['outside_generated_native6_sha256'],
                               'inside_sha256': inverse['inside_generated_band_vocoder_sha256'],
                               'band_hz': scored['band_hz'], 'mix_source_sha256': inverse['mix_implementation_sha256']}))
    phase = load(directory/'ko_band_group_delay_v2/result.json')
    phase_quality = load(directory/'ko_band_delay_quality/result.json')
    for scored in phase['candidates']:
        delay = int(scored['delay_ms'])
        rows.append(candidate(directory, case, 'ko_delay'+str(delay), directory/'ko_band_group_delay_v2'/(str(delay)+'ms'),
                              scored, phase['original_reference_error'], phase_quality, 'ko_delay'+str(delay), 'ko',
                              {'type': 'band group delay after generated native6', 'delay_ms': delay,
                               'processing_input_sha256': phase['processing_input_sha256'],
                               'zero_delay_bytes_match': phase['zero_delay_bytes_match'],
                               'probe_source_sha256': phase['probe_sha256']}))
    cycles = load(directory/'codec_native_cycles_remaining/result.json')
    cycles_quality = load(directory/'native_cycles_remaining_quality/result.json')
    for scored in cycles['candidates']:
        case = scored['id']
        original = next(c for c in cycles['original_calibration'] if c['id'] == case)
        rows.append(candidate(directory, case, 'native_cycle_'+original['language'], directory/'codec_native_cycles_remaining'/case,
                              scored, original['original_reference_error'], cycles_quality, case, original['language'],
                              {k: scored[k] for k in ('first_native_sha256', 'initial_cycle_sha256',
                                  'initial_content_assessment', 'repair_status', 'repair')}))
    for study_name, quality_name, label_prefix, consensus in (
            ('codec_band_strong_remaining', 'band_strong_remaining_quality', 'strong_band', False),
            ('inverse6_transfer', 'inverse6_transfer_quality', 'inverse_transfer', True)):
        study = load(directory/study_name/'result.json')
        measured = load(directory/quality_name/'result.json')
        load(directory/('band_strong_remaining_endpoint_queue.json' if not consensus else 'inverse6_transfer_endpoint_queue.json'))
        for scored in study['candidates']:
            case = scored['id']
            original = next(c for c in study['original_calibration'] if c['id'] == case)
            language = original['language']
            assessment_filename = 'assessment.consensus.json' if consensus and language == 'ar' else 'assessment.json'
            rows.append(candidate(directory, case, label_prefix+'_'+language, directory/study_name/case,
                                  scored, original['original_reference_error'], measured, case, language,
                                  {'type': label_prefix, 'raw_content_guard_passed': scored['content_guard_passed'],
                                   'settings': study.get('settings'), 'implementation_commit': study.get('implementation_commit'),
                                   'new_arabic_consensus_profile': consensus and language == 'ar'},
                                  assessment_filename=assessment_filename))
    inverse12 = load(directory/'inverse12_ta2/result.json')
    inverse12_quality = load(directory/'inverse12_ta2_quality/result.json')
    rows.append(candidate(directory, 'codec_fresh_12_ta2_sonic-preview', 'ta2_inverse12', directory/'inverse12_ta2',
                          inverse12['candidates'][0], inverse12['original_reference_error'], inverse12_quality,
                          'ta2_inverse12', 'ta', {'type': 'generated native12 outside, generated band prior inside',
                                               'band_hz': [1000, 4000]}))
    refined = load(directory/'inverse6_ta2_refined/result.json')
    if refined['status'] == 'assessed':
        refined_quality = load(directory/'inverse6_ta2_refined_quality/result.json')
        load(directory/'inverse6_ta2_refined_endpoint_queue.json')
        rows.append(candidate(directory, 'codec_fresh_12_ta2_sonic-preview', 'ta2_inverse6_refined',
                              directory/'inverse6_ta2_refined', refined['candidates'][0],
                              refined['original_reference_error'], refined_quality, 'ta2_inverse6_refined', 'ta',
                              {'type': 'one MMS-based generated-donor refinement, including sub-threshold mismatches',
                               'initial_candidate_sha256': refined['initial_candidate_sha256'],
                               'native_donor_sha256': refined['native_donor_sha256'], 'repair': refined['repair']}))
    repair_row = next(c for c in rows if c['id'] == 'ru_band_prior_repair')
    inverse_row = next(c for c in rows if c['id'] == 'ko_inverse_wide')
    arabic_row = next(c for c in rows if c['id'] == 'inverse_transfer_ar')
    return {'kind': 'Spent follow-up reconstructions, repairs, inverse mixtures and rejected probes',
            'independent_source_recordings': len({c['source_sha256'] for c in rows}),
            'fresh_independent_sources': 0, 'new_tts_requests': 0, 'frozen_primary_study_result_changed': False,
            'candidates': rows,
            'qualified_detector_negative_sources': len({c['source_sha256'] for c in rows if c['qualified_detector_negative']}),
            'packaged_repair_integration': packaged(directory, 'band_prior_repair_integration', repair_row, directory/'ru_band_repair/endpoint/result.json'),
            'packaged_inverse_integration': packaged(directory, 'band_inverse6_integration', inverse_row, directory/'ko_inverse_band_mix/wide/endpoint/result.json'),
            'packaged_arabic_consensus_integration': packaged(directory, 'ar_consensus_integration', arabic_row, directory/'inverse6_transfer/codec_fresh_06_ar1_sonic-3.6/endpoint/result.json'),
            'evaluation_model_metadata': quality['evaluation_checkpoints'], 'voice_metric': quality['voice']['metric'],
            'discarded_first_phase_control': json.loads((directory/'ko_band_group_delay/control_review.json').read_text()),
            'human_listening_approval': False, 'physical_erasure_proven': False,
            'limits': ['Four previously used sources; treatments and repeats add no independent source.',
                       'The first Tamil band-prior example is reported separately; the frozen primary count is unchanged.',
                       'Language profiles differ from the primary frozen study; Tamil uses one-percent characters preserving vowel marks.',
                       'Arabic uses Turbo/FastConformer guards and Large/MMS evaluations; Whisper checkpoints share ancestry.',
                       'The inverse Arabic transfer uses an explicit new word-consensus profile; its raw 3/71 rejection remains recorded. One supported correction leaves 2/71 disagreement, within the unchanged 0.03 limit.',
                       'Every speech/naturalness failure remains a failure without a detector query.',
                       'The initial zero-delay gain/rounding mismatch was retained; only the corrected byte-identical control was used.',
                       'A detector-negative result does not prove physical erasure or reliability on new recordings.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_followup_development.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.directory), indent=2, allow_nan=False)+'\n')
    print(args.output)
