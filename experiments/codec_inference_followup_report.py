"""Verify codec deferral and secondary-first guard timings against exact files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.vocoder_codec_tradeoff_report import digest, load


def verify_source(report):
    for name, checksum in report['source_files_sha256'].items():
        blob = subprocess.check_output(['git', 'show', report['implementation_commit']+':'+name], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != checksum:
            raise ValueError('Recorded runtime source does not match its Git commit')


def verify_files(directory, study, rows):
    for row in rows:
        folder = directory/study/row['id']
        if (digest(folder/'candidate.flac') != row['canonical_output_sha256'] or
                digest(folder/'metadata.json') != row['metadata_sha256'] or
                digest(directory/'codec_fresh/originals'/(row['source_case']+'.flac')) != row['source_sha256']):
            raise ValueError('A runtime comparison no longer binds its exact source and output')
        metadata = json.loads((folder/'metadata.json').read_text())
        if (not row['content_guard_passed'] or not metadata['content_guard_passed'] or
                metadata['candidate_flac_sha256'] != row['canonical_output_sha256'] or
                metadata['paired_disagreement'] != row['paired_disagreement']):
            raise ValueError('Runtime output and final real-model guards disagree')


def export(directory):
    deferred = load(directory/'lazy_codec_integrations/result.json')
    fast = load(directory/'fast_guard_runtime/result.json')
    profiles = load(directory/'profile_preflight_integrations/result.json')
    reuse = load(directory/'profile_prior_reuse/result.json')
    voice = load(directory/'runtime_voice_preflight/result.json')
    for report in (deferred, fast, profiles, reuse, voice):
        verify_source(report)
        if report['fresh_independent_sources'] != 0 or report['new_http_requests'] != 0 or report['reference_text_used']:
            raise ValueError('These are existing-file runtime comparisons only')
    verify_files(directory, 'lazy_codec_integrations', deferred['candidates'])
    verify_files(directory, 'fast_guard_runtime', fast['candidates'])
    verify_files(directory, 'profile_preflight_integrations', profiles['candidates'])
    if {r['id'] for r in profiles['candidates']} != {'fr','en_long'} or not all(r['exact_previously_checked_file_reproduction'] for r in profiles['candidates']):
        raise ValueError('Retain both real profile preflight integrations')
    if (reuse['original_recognition_inferences'] != 1 or reuse['vocoder_synthesis_calls'] != ['codec-band-prior'] or
            len(reuse['candidates']) != 2 or [r['band_prior_reused'] for r in reuse['candidates']] != [False,True] or
            [r['source_job_reused'] for r in reuse['candidates']] != [False,True]):
        raise ValueError('The fallback must preserve its one original recognition and one shared synthesis')
    for index,row in enumerate(reuse['candidates']):
        folder=directory/'profile_prior_reuse'
        if (digest(folder/(str(index)+'.flac')) != row['canonical_output_sha256'] or
                digest(folder/(str(index)+'.json')) != row['metadata_sha256'] or
                not row['content_guard_passed'] or not row['exact_previously_checked_file_reproduction']):
            raise ValueError('Prior reuse changed its bound whole-file output or guard')
    if [r['decision']['passed'] for r in voice['cases']] != [True,False]:
        raise ValueError('The actual voice gate must retain both its acceptance and known rejection')
    for row in voice['cases']:
        receipt=directory/'runtime_voice_preflight/runtime_voice'/row['id']/(row['decision']['candidate_sha256']+'.json')
        if digest(receipt) != row['decision']['receipt_sha256'] or row['decision']['independent_evaluation']:
            raise ValueError('Voice gates are bound operational estimates, not independent validation')
    calls = {r['id']: [c['bandwidth_kbps'] for c in r['native_codec_calls']] for r in deferred['candidates']}
    if calls != {'ta1': [], 'ko': [6.], 'ru': [12.], 'ar': [6.]}:
        raise ValueError('A required donor disappeared or an unused pass was recomputed')
    if {r['id'] for r in fast['candidates']} != {'ru', 'ko'}:
        raise ValueError('Retain both the repaired-rejection and accepted-first comparisons')
    for row in fast['candidates']:
        previous = next(c for c in deferred['candidates'] if c['id'] == row['id'])
        if (previous['source_sha256'] != row['source_sha256'] or
                previous['canonical_output_sha256'] != row['canonical_output_sha256'] or
                row['previous_same_output_elapsed_seconds'] != previous['elapsed_seconds'] or
                not row['both_final_recognizers_ran']):
            raise ValueError('The final exact-file comparison changed speech guard requirements')
    rejected = fast['warm_rejected_candidate_comparison']
    if (rejected['full_assessment']['content_guard_passed'] or rejected['early_assessment']['content_guard_passed'] or
            rejected['early_assessment']['paired_disagreement']['turbo'] is not None or
            rejected['full_assessment']['paired_disagreement']['mms'] != rejected['early_assessment']['paired_disagreement']['mms'] or
            not rejected['same_secondary_units_and_alignment'] or not rejected['primary_inference_skipped']):
        raise ValueError('The skipped inference must remain a rejected candidate')
    # The old private field called every native timing unused, including the RU
    # donor that is required for repair. Publish the accurate role explicitly.
    for row in deferred['candidates']:
        row['previous_native12_seconds'] = row.pop('previous_unused_native12_seconds')
        row['previous_native12_unused'] = row['id'] != 'ru'
    return {'kind': 'Existing-file codec inference optimizations', 'fresh_independent_sources': 0,
            'new_http_requests': 0, 'new_tts_requests': 0, 'frozen_primary_study_result_changed': False,
            'deferred_native_integration': deferred, 'secondary_first_guard_integration': fast,
            'current_profile_preflight_integrations': profiles,
            'original_and_prior_reuse_integration': reuse, 'runtime_voice_gate_preflight': voice,
            'human_listening_approval': False, 'physical_erasure_proven': False,
            'limits': ['Four codec-deferral files and two secondary-first files reuse previously detector-checked outputs exactly.',
                       'The unused native12 passes took under one second each in these earlier runs; removing them did not establish an end-to-end speedup.',
                       'Secondary-first rejection skips an expensive primary only after the other recognizer fails; both final recognizers remain mandatory for accepted output.',
                       'A skipped primary has a null disagreement, never a passing score or an imputed measurement.',
                       'Timings exclude outer runtime construction and come from individual sequential CPU runs, without statistical speed guarantees.',
                       'Original/prior reuse times original recognition separately; the first and fallback methods share warm models and one vocoder synthesis, not free original calibration.',
                       'The voice-gate preflight accepts a previously qualified file and rejects a known naturalness failure using the same fixed thresholds; it adds no fresh source.',
                       'No new removal efficacy or fresh independent recordings are established by inference parity.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_inference_followup.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.directory), indent=2, allow_nan=False)+'\n')
    print(args.output)
