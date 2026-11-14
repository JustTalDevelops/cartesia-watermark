"""Seal the authored 48-request profile study without making service calls."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = (
    'tools/verify_reconstruction.py', 'tools/reconstruction_worker.py',
    'tools/codec_settings.py', 'tools/codec_profiles.py', 'tools/codec_assets.json',
    'tools/codec_audio.py', 'tools/codec_inference.py', 'tools/codec_teacher.py',
    'tools/codec_content.py', 'tools/codec_text.py', 'tools/codec_repair.py',
    'tools/codec_word_consensus.py', 'tools/korean_spacing.py', 'tools/speech_repair.py',
    'tools/speech_repair_assets.json', 'tools/whisper_encoder_reuse.py',
    'tools/reconstruct_codec.py', 'tools/reconstruct_codec_band.py', 'tools/vocoder_band_prior.py',
    'tools/reconstruct_vocoder.py', 'tools/vocoder_inference.py', 'tools/vocoder_assets.json',
    'tools/cartesia.py', 'tools/eval_voice_quality.py', 'tools/remover.py', 'tools/quality_blend.py',
    'experiments/generate_codec_study.py', 'experiments/run_codec_study.py',
    'experiments/run_codec_profiles.py', 'experiments/codec_profile_cache.py',
    'experiments/codec_profile_evaluation.py', 'experiments/score_codec_study.py',
    'experiments/codec_evaluation_text.py', 'experiments/codec_evaluation_assets.json',
    'experiments/freeze_codec_profiles.py',
    'requirements/codec.txt', 'requirements/speech_repair.txt', 'requirements/verification.txt',
    'requirements/vocoder.txt', 'requirements/codec_evaluation.txt', 'requirements/eval.txt',
    'requirements/inference.txt',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(plan_path, output, directory):
    plan = json.loads(plan_path.read_text())
    marked = [i for i in plan['items'] if i['marked_provenance']]
    controls = [i for i in plan['items'] if not i['marked_provenance']]
    if (len(plan['items']) != 48 or len(marked) != 44 or len(controls) != 4 or
            len({i['id'] for i in plan['items']}) != 48 or len({i['voice'] for i in marked}) != 44 or
            len({i['text'] for i in marked}) != 44 or
            Counter(i['language'] for i in marked) != Counter({l: 4 for l in ('en','es','fr','de','hi','ja','ar','ko','zh','ru','ta')}) or
            any(Counter(i['model'] for i in marked if i['language'] == language) !=
                Counter({'sonic-preview': 2, 'sonic-3.6': 2}) for language in {i['language'] for i in marked})):
        raise ValueError('Retain the entire predeclared balanced 44-marked/four-control cohort')
    for control in controls:
        match = next(i for i in marked if i['id'] == control['matched_marked_request'])
        if control['model'] != 'sonic-3.5' or any(control[k] != match[k] for k in
                ('voice', 'language', 'text', 'stereo_derivation_gains')):
            raise ValueError('Nominal controls must retain their matched voice, text and format')
    if any(i['sample_rate'] != 44100 or i['container'] != 'wav' or i['encoding'] != 'pcm_f32le' for i in plan['items']):
        raise ValueError('Use the predeclared whole-provider format')
    budget_path = directory/'budget.json'
    if budget_path.is_symlink():
        raise ValueError('Use the original regular budget ledger, not an alias file')
    budget = json.loads(budget_path.read_text())
    extra = sum(2*len(i['text'].encode('utf8')) for i in plan['items'])
    if (budget['authorized_cap_usd'] != 50 or budget['reservation_rate_usd_per_million_credits'] != 65 or
            sum(r['reserved_credits'] for r in budget['requests']) != budget['reserved_credits'] or
            budget['reserved_credits']+extra > min(budget['maximum_reserved_credits'], int(50e6/65)) or
            any(i['id'] in {r['id'] for r in budget['requests']} for i in plan['items']) or
            (directory/'codec_profiles_fresh/generation.json').exists()):
        raise ValueError('Budget integrity, shared cap or prospective freshness check failed')
    projection = plan['budget_projection']
    if (projection['previous_reserved_credits'] != budget['reserved_credits'] or
            projection['additional_conservative_credits'] != extra or
            projection['reservation_rate_usd_per_million_credits'] != 65 or
            abs(projection['combined_reserve_usd']-(budget['reserved_credits']+extra)*65/1e6) > 1e-9):
        raise ValueError('Budget projection no longer matches the original shared ledger')
    availability_path = directory/'codec_profiles_fresh.voice_availability.live.json'
    availability = json.loads(availability_path.read_text())
    if (not availability['complete'] or availability['new_tts_requests'] != 0 or
            {v['id'] for v in availability['voices']} != {i['voice'] for i in marked} or
            digest(availability_path) != plan['voice_selection']['availability_receipt_sha256']):
        raise ValueError('Current availability must cover every selected public voice')
    if any(v['is_owner'] or v['is_pro'] or v['status'] != 'active' or v['visibility'] not in ('all', 'public')
           for v in availability['voices']):
        raise ValueError('The availability receipt contains an ineligible voice')
    for name in ('profile_preflight_integrations', 'profile_prior_reuse', 'runtime_voice_preflight'):
        if json.loads((directory/name/'result.json').read_text())['complete'] is not True:
            raise ValueError('Complete actual-model development preflights before freezing')
    model_check = json.loads((directory/'codec_profiles_model_check.json').read_text())
    if model_check.get('ready') is not True:
        raise ValueError('Pinned local models must pass their readiness check')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    hashes = {name: digest(ROOT/name) for name in SOURCE_FILES}
    for name, checksum in hashes.items():
        blob = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != checksum:
            raise ValueError('Commit every source/evaluation file before sealing')
    if plan['candidate_implementation_commit'] != commit or plan['source_files_sha256'] != hashes:
        raise ValueError('Prepared plan must identify the committed implementation being frozen')
    suites = {}
    for name in ('codec_profiles_offline_tests.log', 'codec_profiles_extra_tests.log'):
        text = (directory/name).read_text()
        match = re.search(r'Ran (\d+) tests? in ', text)
        if match is None or not text.rstrip().endswith('OK'):
            raise ValueError('Finish all required offline suites before sealing')
        suites[name] = {'passed': int(match[1]), 'log_sha256': digest(directory/name)}
    report = {'status': 'Prospective freeze; no efficacy result', 'sealed_at': datetime.now(timezone.utc).isoformat(),
              'generation_plan_path': str(plan_path.resolve().relative_to(ROOT)),
              'generation_plan_sha256': digest(plan_path), 'candidate_implementation_commit': commit,
              'evaluation_implementation_commit': commit, 'source_files_sha256': hashes,
              'fresh_generation_started': False, 'fresh_candidate_detector_labels_observed': False,
              'planned_requests': 48, 'marked_provenance_requests': 44, 'matched_nominal_controls': 4,
              'negative_confirmations': 2, 'positive_counterexample_confirmations': 2,
              'original_confirmations': 2, 'maximum_http_attempts_per_phase': {'baseline': 96, 'candidates': 292},
              'minimum_detector_start_interval_seconds': 80,
              'selection_policy': 'Fixed codec_profiles order from detected language and complete duration; both processing ASRs, fixed speaker/naturalness estimates and detector responses only. No reference text or evaluation-ASR scores select methods. Plan text is checked only for integrity.',
              'endpoint_policy': 'Direct route and shared pacing; whole exact files; stop on errors or conflicting repeats; no rotation or automatic observation replay; duplicate checked candidates skipped.',
              'quality_implementation': 'experiments/codec_profile_evaluation.py:quality_criteria',
              'quality_rules': ['All 48 planned requests remain in results, including rejected, unsupported, error, undetected-original and insufficient-calibration cases.',
                  'Complete original-only processing/evaluation ASR and voice calibration before candidate selection.',
                  'Hindi/Tamil retain essential vowel signs and viramas: one-percent paired and reference-character-delta limits; other profiles use three percent with the declared character/word/spacing policy.',
                  'Primary and secondary guards are not independent evaluations. Hindi evaluation uses Turbo transcription, Arabic uses Large plus MMS, other languages use Small; Whisper checkpoints share ancestry.',
                  'Require at least two original-calibrated checkpoints: reference error at most 0.10 for Hindi/Tamil, 0.15 otherwise; wrong detected language remains unproven.',
                  'Speaker minimum 0.95 and DNSMOS overall delta at least -0.2 are operational selection gates, not independent quality evidence or human approval.',
                  'Independent evaluation-ASR reference deltas are graded after selection; failed selected files remain failures without a quality-triggered alternate selection.',
                  'A marked source must be measured positive before a twice-negative output can be counted as a transition; nominal controls and repeated queries are not additional removal successes.',
                  'Four matched controls reuse four marked voices/texts. There are 44 distinct marked voice IDs, not 48 independent voices. Some requested utterance languages differ from public primary-language tags.',
                  'Preserve whole sample rate, channel count and sample count; no detector padding/cropping; stereo derivatives use predeclared gains.',
                  'No local-detector score enters selection. No human listening approval or physical erasure guarantee is claimed.'],
              'budget': dict(plan['budget_projection'], ledger='budget.json', generation_subdirectory='codec_profiles_fresh'),
              'offline_suites': suites, 'actual_model_preflight_receipts_sha256': {
                  name: digest(directory/name/'result.json') for name in
                  ('profile_preflight_integrations', 'profile_prior_reuse', 'runtime_voice_preflight')},
              'model_readiness_receipt_sha256': digest(directory/'codec_profiles_model_check.json'),
              'human_listening_approval': False, 'physical_erasure_proven': False}
    if output.exists():
        raise ValueError('Prospective freezes are not overwritten')
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    args = parser.parse_args()
    report = seal(args.plan, args.output, args.directory)
    print('PROFILE_FREEZE_CREATED', report['planned_requests'], 'requests', report['budget']['combined_reserve_usd'], 'conservative combined USD reserve')
