"""Export checked metadata for Whisper memory and packaged-audio parity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def build(directory, implementation_commit, language_fix_commit):
    folder = directory/'whisper_memory_probe'
    summary = load(folder/'summary.json')
    baseline = load(directory/'speech_repair_long_preflight.baseline_incomplete.json')
    automatic = load(directory/'speech_repair_long_preflight.json')
    before_language_fix = load(directory/'speech_repair_long_preflight.pre_language_fix.json')
    package = load(directory/'memory_optimized_package_smoke/result.json')
    require(summary['complete'] is True and package['complete'] is True and len(package['cases']) == 2,
            'Incomplete inference or package probes')
    require(len(summary['cases']) == 2 and all(r['all_text_word_timestamps_and_language_exact'] is True
            for r in summary['cases']), 'Recognition parity failed')
    for case in summary['cases']:
        before, after = [load(folder/(case['case']+'_'+mode+'.json')) for mode in ('baseline', 'optimized')]
        require(before['source_sha256'] == after['source_sha256'] == case['source_sha256'] and
                before['result'] == after['result'] and before['language'] == after['language'],
                'Recognition parity differs from its actual output')
    long = load(folder/'english252_optimized.json')
    require(long['source_sha256'] == baseline['source_sha256'] == summary['long_probe']['source_sha256'] and
            baseline['recognition_completed'] is False and long['encoder_hooks'] == 32,
            'Long probe changed input or baseline completion status')
    require(summary['implementation_sha256']['memory_helper'] == digest(ROOT/'tools/whisper_memory.py'),
            'Memory helper changed after recognition measurements')
    frozen_recognizer = subprocess.check_output(
        ['git', 'show', '05c244c:tools/reconstruct_speech_repair.py'], cwd=ROOT)
    require(hashlib.sha256(frozen_recognizer).hexdigest() == summary['implementation_sha256']['recognizer'],
            'Original recognition probe source differs from the cited baseline')
    require(automatic['longform_completed'] is True and automatic['language'] == 'en' and
            automatic['chunks_cover_recognized_text'] is True and automatic['invalid_timestamp_chunks'] == 0 and
            automatic['original_language_automatically_inferred'] is True
            and automatic['source_sha256'] == baseline['source_sha256'] and
            automatic['recognizer_code_sha256'] == digest(ROOT/'tools/reconstruct_speech_repair.py') and
            automatic['recognizer']['encoder_attention_diagnostics_retained'] is False,
            'Production automatic-language long-source preflight did not complete')
    commit = subprocess.check_output(['git', 'rev-parse', implementation_commit], cwd=ROOT, text=True).strip()
    for name, expected in package['code_sha256'].items():
        frozen = subprocess.check_output(['git', 'show', commit+':tools/'+name], cwd=ROOT)
        require(hashlib.sha256(frozen).hexdigest() == expected,
                'Packaged implementation differs from its cited commit')
    language_commit = subprocess.check_output(['git', 'rev-parse', language_fix_commit], cwd=ROOT, text=True).strip()
    language_sources = {}
    for name in ('reconstruct_speech_repair.py', 'speech_repair.py'):
        frozen = subprocess.check_output(['git', 'show', language_commit+':tools/'+name], cwd=ROOT)
        require(hashlib.sha256(frozen).hexdigest() == digest(ROOT/'tools'/name), 'Language fix changed after its commit')
        language_sources[name] = digest(ROOT/'tools'/name)
    require(before_language_fix['language'] is None and before_language_fix['source_sha256'] == automatic['source_sha256'],
            'Missing original long-form language counterexample')
    for case in package['cases']:
        output = directory/'memory_optimized_package_smoke'/(case['id']+'.flac')
        original = directory/'word_repair_vocoder_dev'/(case['id']+'__donor100.flac')
        require(digest(output) == digest(original) == case['output_sha256'] and
                case['canonical_flac_bytes_exact_to_baseline'] is True and
                case['encoder_diagnostics_retained'] is False and case['quality_verified'] is False,
                'Packaged audio parity or its quality interpretation changed')
        before, rate = sf.read(original, dtype='float32')
        after, actual_rate = sf.read(output, dtype='float32')
        require(actual_rate == rate and np.array_equal(before, after), 'Packaged waveform parity failed')
    require(package['parent_without_torch'] is True, 'Verification parent unexpectedly imported Torch')
    return {'schema': 1, 'implementation_commit': commit,
            'scope': 'Memory optimization and output-equivalence probes; zero fresh efficacy recordings',
            'change': 'Discard unused encoder self-attention diagnostics after each layer computes its output; decoder cross attention for word alignment is retained.',
            'recognition_parity': summary['cases'], 'optimized_whole_long_recording': summary['long_probe'],
            'recognition_probe_source_sha256': summary['implementation_sha256'],
            'recognition_probe_setup': 'Baseline recognizer from 05c244c; attach the same encoder-memory helper for optimized comparisons. The packaged probe exercises the integrated constructor.',
            'optimized_whole_long_language_policy': 'English forced for this timing probe; automatic original-language inference is checked separately below.',
            'automatic_language_whole_long_preflight': automatic,
            'longform_language_fix': {'implementation_commit': language_commit, 'source_sha256': language_sources,
                                     'before': before_language_fix,
                                     'result': 'Missing long-form chunk tags now invoke the pinned model initial-feature language detector. Conflicting or unknown tags remain rejected.'},
            'optimized_long_word_chunks': summary['long_word_chunks'],
            'incomplete_long_baseline': baseline,
            'packaged_whole_recording_parity': package['cases'],
            'verification_parent_without_torch': True, 'source_sha256': package['code_sha256'],
            'limits': [summary['limits'],
                       'English60 is an analysis-only performance fixture derived from a longer generated source; it is never a detector input.',
                       'The 252-second baseline was stopped after excessive memory retention; no complete baseline transcript or runtime was inferred.',
                       'Only the 28-second and 60-second fixtures have complete baseline/optimized transcript and timestamp comparisons.',
                       'Short-case recognition latency was approximately unchanged.',
                       'Packaging repeats spent Arabic/Tamil fixtures and adds no efficacy samples or new detector observations.',
                       'Removal and Arabic content-fidelity failures remain unresolved by this memory optimization.'],
            'transcripts_exported': False, 'private_recordings_included': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/speech_repair_runtime.json')
    parser.add_argument('--implementation-commit', default='9d0d4ad')
    parser.add_argument('--language-fix-commit', default='HEAD')
    args = parser.parse_args()
    report = build(args.directory, args.implementation_commit, args.language_fix_commit)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Exported two exact recognition comparisons, complete long probe and two exact packaged outputs')


if __name__ == '__main__':
    main()
