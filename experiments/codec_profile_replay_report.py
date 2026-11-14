"""Reassess saved selected files without rewriting their frozen study."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.vocoder_codec_tradeoff_report import digest, load


def export(directory):
    study = load(directory/'selected_profiles_replay/result.json')
    quality = load(directory/'selected_profiles_replay_quality/result.json')
    primary_path = ROOT/'results/codec_reliability.json'
    primary = json.loads(primary_path.read_text())
    if digest(primary_path) != study['primary_report_sha256']:
        raise ValueError('The frozen source report changed')
    rows = []
    for assessed in study['candidates']:
        case = assessed['id']
        previous = next(c for c in primary['clips'] if c['id'] == case)
        source = directory/'codec_fresh/originals'/(case+'.flac')
        output = directory/'selected_profiles_replay'/case/'candidate.flac'
        if (digest(source) != assessed['source_sha256'] or digest(output) != assessed['output_sha256'] or
                assessed['output_sha256'] != previous['output_sha256'] or
                previous['negative_checks'] != 2):
            raise ValueError('A replay no longer binds its exact previously checked file')
        original = next(c for c in study['original_calibration'] if c['id'] == case)
        evaluation = [c for c in quality['candidates'] if c['id'] == case]
        primary_eval = next(c for c in evaluation if c['evaluation_checkpoint'] != 'mms')
        voice = next(c for c in quality['voice']['clips'] if c['id'] == case)
        if any(c['source_sha256'] != assessed['source_sha256'] or c['output_sha256'] != assessed['output_sha256']
               for c in evaluation+[voice]):
            raise ValueError('Quality estimates refer to a different complete file')
        maximum = assessed['maximum_paired_disagreement']
        p, s = assessed['primary_guard'], assessed['secondary_guard']
        gates = {'primary_paired': assessed['paired_disagreement'][p] <= maximum+1e-12,
                 'secondary_paired': assessed['paired_disagreement'][s] <= maximum+1e-12,
                 'primary_reference_delta': assessed['reference_error_delta_evaluation_only'][p] <= maximum+1e-12,
                 'secondary_reference_delta': assessed['reference_error_delta_evaluation_only'][s] <= maximum+1e-12,
                 'evaluation_reference_delta': primary_eval['reference_error_delta'] <= maximum+1e-12,
                 'speaker_minimum': voice['speaker_minimum'] >= .95,
                 'dnsmos_overall': voice['dnsmos_delta']['overall'] >= -.2}
        if assessed['language'] == 'ar':
            extra = next(c for c in evaluation if c['evaluation_checkpoint'] == 'mms')
            gates['additional_mms_reference_delta'] = extra['reference_error_delta'] <= maximum+1e-12
        errors = list(original['original_reference_error'].values())
        errors.extend(c['original_reference_error'] for c in quality['original_calibration'] if c['id'] == case)
        ceiling = .1 if assessed['language'] in ('hi', 'ta') else .15
        calibrated = sum(e <= ceiling for e in errors)
        rows.append({'id': case, 'source_sha256': assessed['source_sha256'],
                     'candidate_sha256': assessed['output_sha256'], 'language': assessed['language'],
                     'marked_provenance': previous['marked_provenance'],
                     'historical_primary_qualified': previous['qualified_checked_output'],
                     'historical_negative_checks': previous['negative_checks'],
                     'historical_original_detected': previous['original_detected'],
                     'reassessed_content': assessed, 'independent_evaluation': evaluation, 'voice': voice,
                     'quality_gate_results': gates, 'calibrated_original_checkpoints': calibrated,
                     'current_profile_quality_passed': assessed['content_guard_passed'] and all(gates.values()) and calibrated >= 2})
    if len(rows) != 9 or sum(not r['marked_provenance'] for r in rows) != 2:
        raise ValueError('Retain all seven marked selected files and both nominal controls')
    return {'kind': 'Compatibility replay of nine saved selected files',
            'independent_source_recordings': 9, 'fresh_independent_sources': 0,
            'new_neural_synthesis': False, 'new_http_requests': 0,
            'frozen_primary_study_result_changed': False, 'frozen_primary_qualified_marked_count': 6,
            'recognizer_implementation_commit': study['recognizer_implementation_commit'],
            'primary_report_sha256': study['primary_report_sha256'], 'candidates': rows,
            'current_profile_quality_passed': sum(r['current_profile_quality_passed'] for r in rows),
            'marked_files_passing_current_profile': sum(r['marked_provenance'] and r['current_profile_quality_passed'] for r in rows),
            'nominal_controls_passing_current_profile': sum(not r['marked_provenance'] and r['current_profile_quality_passed'] for r in rows),
            'evaluation_metadata': quality['evaluation_checkpoints'], 'voice_metric': quality['voice']['metric'],
            'human_listening_approval': False, 'physical_erasure_proven': False,
            'limits': ['Every file was selected and checked previously; old exact-file detector receipts are reused, not new observations.',
                       'Hindi uses the prospective vowel-preserving one-percent character policy and Turbo evaluation, replacing neither its old failure nor the frozen count.',
                       'Arabic uses the additional Large/MMS evaluations; Whisper checkpoints share ancestry.',
                       'The two nominal controls are not watermark-removal transitions.',
                       'This compatibility replay adds no held-out removal evidence.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_profile_replay.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.directory), indent=2, allow_nan=False)+'\n')
    print(args.output)
