"""Curate exploratory ASR timings with source revisions and failed attempts.

Does not turn repetitions into independent samples or claim removal efficacy.
Input reports contain counts and timing/parity results, not recognized text.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(path.read_text())


def revision(name):
    return subprocess.check_output(['git', 'rev-parse', name], cwd=ROOT, text=True).strip()


def check_source(commit, name, expected):
    data = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Measured source differs from its recorded Git revision')


def export(directory):
    freeze_path = ROOT/'results/codec_reliability.freeze.json'
    freeze = load(freeze_path)
    for name, checksum in freeze['source_files_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('The claimed unchanged processing/evaluation snapshot has changed')
    trials = []
    for folder, ref in [('codec_asr_quantization', '88e90ce'), ('codec_asr_precision_modelonly', '4a16c30')]:
        report = load(directory/folder/'result.json')
        commit = revision(ref)
        check_source(commit, 'experiments/benchmark_codec_asr.py', report['source_sha256'])
        if not report['complete'] or report['fresh_sources_read'] != 0 or report['api_calls'] != 0:
            raise ValueError('Expected completed spent-source benchmarks without API requests')
        if len({r['input_sha256'] for r in report['cases']}) != 4:
            raise ValueError('Benchmark treatment count changed')
        summary = []
        for metric in ('turbo', 'mms'):
            baseline = [r for r in report['cases'] if r['metric'] == metric and r['method'] == 'fp32']
            baseline_mean = statistics.mean(t for r in baseline for t in r['times_seconds'])
            for method in report['methods']:
                rows = [r for r in report['cases'] if r['metric'] == metric and r['method'] == method]
                if len(rows) != 4 or any(len(r['times_seconds']) != 2 for r in rows):
                    raise ValueError('All four treatments and both repetitions must remain')
                times = [t for r in rows for t in r['times_seconds']]
                if not all(math.isfinite(t) and t > 0 for t in times):
                    raise ValueError('Invalid timing observation')
                mean = statistics.mean(times)
                summary.append({'metric': metric, 'method': method, 'mean_inference_seconds': mean,
                                'baseline_relative_speed': baseline_mean/mean,
                                'normalized_text_identical_files': sum(all(r['normalized_units_identical_to_fp32']) for r in rows),
                                'tested_files': 4})
        trials.append({'source_commit': commit, 'observations': report, 'summary': summary})
    failed = load(directory/'codec_asr_precision/result.json')
    failed_commit = revision('f7b5b46')
    check_source(failed_commit, 'experiments/benchmark_codec_asr.py', failed['source_sha256'])
    if failed['complete']:
        raise ValueError('Expected preserved failed whole-call BF16 attempt')
    encoder = load(directory/'codec_encoder_reuse/result.json')
    encoder_commit = revision('6a99311')
    check_source(encoder_commit, 'experiments/benchmark_whisper_encoder_reuse.py', encoder['benchmark_source_sha256'])
    check_source(encoder_commit, 'experiments/whisper_encoder_reuse.py', encoder['helper_source_sha256'])
    if not encoder['complete'] or len(encoder['cases']) != 8:
        raise ValueError('Complete four-treatment encoder comparison required')
    original_precision = load(directory/'mms_bf16_originals/result.json')
    precision_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
                                               'experiments/benchmark_mms_bf16_originals.py'], cwd=ROOT, text=True).strip()
    check_source(precision_commit, 'experiments/benchmark_mms_bf16_originals.py', original_precision['source_sha256'])
    if not original_precision['complete'] or len(original_precision['clips']) != 16 or original_precision['candidate_outputs_read']:
        raise ValueError('Expected all sixteen original-only precision observations')
    return {'scope': 'Exploratory local CPU recognition timings and parity; not a removal efficacy study',
            'unchanged_processing_freeze_sha256': hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            'machine': {'cpu': 'AMD Ryzen9 9900X', 'cpu_quota_cores': 4, 'neural_jobs_concurrent': 1},
            'independent_spent_sources_in_paired_timings': 2, 'audio_treatments_per_model': 4,
            'new_independent_removal_samples': 0, 'processing_runtime_modified': False,
            'precision_trials': trials,
            'failed_attempt': {'source_commit': failed_commit, 'observations': failed,
                               'error_type': 'TypeError', 'error': 'Got unsupported ScalarType BFloat16',
                               'cause': 'Autocast included Whisper preprocessing, whose NumPy conversion does not accept BF16.',
                               'correction': 'Restrict autocast to neural generate/forward; retain FP32 preprocessing.'},
            'encoder_reuse': {'source_commit': encoder_commit, 'observations': encoder,
                              'all_tested_tokens_and_languages_identical': all(all(r['exact_token_parity']) and all(r['language_parity']) for r in encoder['cases']),
                              'deployed_in_frozen_runtime': False},
            'multilingual_original_only_mms': {'source_commit': precision_commit, 'observations': original_precision,
                'exact_normalized_text_parity_count': sum(r['exact_normalized_unit_parity'] for r in original_precision['clips']),
                'frozen_quality_rules_or_results_changed': False},
            'limitations': ['Timings exclude model loading/conversion; repeated calls are not independent recordings.',
                'Both paired timing sources were spent development inputs. No human listening approval or erasure claim.',
                'INT8 MMS tests are fixed-language; they do not implement multilingual adapter switching after quantization.',
                'Preserved text on these files does not prove global equivalence. Precision alternatives remain prototypes.',
                'Original-only multilingual comparison reads fresh originals, not fresh candidate files; adds no removal evidence.',
                'Encoder reuse is restricted to short inputs; the complete long-input path remains unchanged.'],
            'primary_references': ['https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_dynamic.html',
                'https://docs.pytorch.org/docs/stable/amp.html',
                'https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/whisper/generation_whisper.py']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_asr_runtime.json')
    args = parser.parse_args()
    result = export(args.directory)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
