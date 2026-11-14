"""Validate and export exact-file codec runtime integration on two spent sources.

Requires the implementation commit containing the tested files. No recordings,
recognized words, speaker embeddings, or private recording fingerprints export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'tools'))
from speech_repair import pcm_digest


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_commit(commit, sources):
    require(re.fullmatch('[0-9a-f]{40,64}', commit), 'Supply the complete implementation commit hash')
    for name, expected in sources.items():
        data = subprocess.check_output(['git', 'show', commit+':tools/'+name], cwd=ROOT)
        require(hashlib.sha256(data).hexdigest() == expected, 'Implementation commit differs from tested '+name)


def build(directory, commit):
    folder = directory/'codec_runtime_verified'
    protocol_path = folder/'protocol.json'
    protocol, report = load(protocol_path), load(folder/'result.json')
    require(report['complete'] is True and report['protocol_sha256'] == digest(protocol_path) and
            len(report['cases']) == 2 and len(report['http_attempts']) == len(report['checks']) == 6 and
            all(r['http_status'] == 200 for r in report['http_attempts']), 'Incomplete controlled CLI validation')
    require(protocol['presets'] == ['codec-native', 'codec-band'] and protocol['negative_confirmations'] == 2,
            'Unexpected method order or confirmation count')
    canonical = {r['id']: r for r in load(directory/'source_encoding_dev/protocol.json')['items']}
    controls = [r for r in report['checks'] if r['role'].endswith('positive_control')]
    require(len(controls) == 2 and all(r['detected'] is True and r['http_status'] == 200 and
            r['sha256'] == canonical[r['id']]['output_sha256'] == digest(ROOT/canonical[r['id']]['output'])
            for r in controls), 'Whole-source positive controls failed')
    development_path = ROOT/'results/reconstruction_codec_development.json'
    development = load(development_path)
    quality = {r['output_sha256']: r for name in ['vocos_encodec', 'codec_vocoder_band_mixtures']
               for r in development[name]['cases']}
    sources, cases = dict(protocol['source_files_sha256']), []
    planned = {r['id']: r for r in protocol['items']}
    for measured in report['cases']:
        key = measured['id']
        plan, audit = planned[key], load(folder/(key+'.audit.json'))
        output = folder/(key+'.flac')
        expected = plan['expected_selected_flac_sha256']
        require(audit['status'] == 'verified_negative' and audit['source_sha256'] == plan['source_sha256'] ==
                measured['source_sha256'] and audit['output_sha256'] == measured['output_sha256'] == expected == digest(output) and
                audit['selected_preset'] == measured['selected_preset'] == plan['expected_selected_preset'] and
                audit['negative_checks'] == 2 and audit['quality_verified'] is False and
                measured['exact_development_flac_parity'] is True and measured['torch_or_scipy_imported_in_parent'] is False,
                'CLI result does not match the planned exact file or parent environment')
        require(audit['verification_cli_code_sha256'] == sources['verify_reconstruction.py'] and
                audit['reconstructor_code_sha256'] == sources['reconstruct_codec.py'], 'Wrong CLI or reconstruction implementation')
        for name, checksum in audit['source_files_sha256_at_start'].items():
            require(name not in sources or sources[name] == checksum, 'Conflicting runtime source fingerprints')
            sources[name] = checksum
        samples, rate = sf.read(output, dtype='float32')
        require(rate == audit['sample_rate'] and len(samples) == audit['samples'] and
                (1 if samples.ndim == 1 else samples.shape[1]) == audit['channels'] and np.isfinite(samples).all() and
                np.abs(samples).max() < 1 and np.array_equal(samples*2**23, np.rint(samples*2**23)),
                'Published recording changed format, clipped, or left the assessed PCM24 grid')
        selected = audit['attempts'][-1]
        model = selected['model']
        require(selected['state'] == 'checked' and selected['sha256'] == expected and
                model['content_guard_passed'] is True and model['quality_verified'] is False and
                model['candidate_flac_sha256'] == expected and model['content_guard_pcm_sha256'] == pcm_digest(samples) and
                model['content_guard_sample_grid'] == 'PCM_24' and
                all(v is not None and v <= .03+1e-12 for v in model['paired_disagreement'].values()),
                'Published samples differ from the content-assessed recording')
        checks = selected['checks']
        require([r['ordinal'] for r in checks] == [1, 2] and
                all(r['state'] == 'checked' and r['http_status'] == 200 and r['detected'] is False for r in checks),
                'Missing two actual verifier confirmations')
        ledger = [r for r in report['checks'] if r['id'] == key and r['role'].startswith('processed_')]
        require(len(ledger) == 2 and all(r['sha256'] == expected and r['http_status'] == 200 and r['detected'] is False
                for r in ledger), 'Supplemental HTTP ledger does not match verifier confirmations')
        estimates = quality[expected]
        require(all(estimates['estimated_quality_criteria'].values()) and estimates['source_sha256'] == plan['source_sha256'],
                'Quality estimates belong to another development file or failed a fixed threshold')
        if model['preset'] == 'codec-band':
            require(len(audit['attempts']) == 2 and audit['attempts'][0]['state'] == 'content_rejected' and
                    audit['attempts'][0]['model']['content_guard_passed'] is False and model['source_job_reused'] is True,
                    'Arabic fallback bypassed content rejection or lost immediate source reuse')
        else:
            require(model['preset'] == 'codec-native' and len(audit['attempts']) == 1, 'Unexpected native workflow')
        cases.append({'id': key, 'source_sha256': plan['source_sha256'], 'output_sha256': expected,
                      'selected_preset': model['preset'], 'sample_rate': rate, 'samples': len(samples),
                      'channels': audit['channels'], 'exact_development_flac_parity': True,
                      'actual_cli_negative_confirmations': checks, 'supplemental_ledger_role_labels': [r['role'] for r in ledger],
                      'content_assessed_pcm_sha256': model['content_guard_pcm_sha256'],
                      'estimated_quality_reused_from_identical_development_file': estimates['quality_estimates'],
                      'all_seven_estimated_quality_criteria_passed': True,
                      'attempts': [{'preset': r['preset'], 'state': r['state'],
                                    'content_guard_passed': r['model']['content_guard_passed'],
                                    'paired_disagreement': r['model']['paired_disagreement'],
                                    'worker_job_seconds_excluding_initial_model_load_and_api_waits': r['model']['elapsed_seconds'],
                                    'source_job_reused': r['model']['source_job_reused']} for r in audit['attempts']],
                      'torch_or_scipy_imported_in_verification_parent': False,
                      'quality_verified': False})
    parity_folder = directory/'codec_content_parity'
    parity_protocol, parity = load(parity_folder/'protocol.json'), load(parity_folder/'result.json')
    require(parity['complete'] is True and parity['protocol_sha256'] == digest(parity_folder/'protocol.json') and
            len(parity['cases']) == 3 and all(r['turbo_units_exact_reference'] is True and r['mms_units_exact_reference'] is True
            for r in parity['cases']), 'Recognizer unit parity is incomplete')
    require(all(sources.get(name) == checksum for name, checksum in parity_protocol['source_files_sha256'].items()),
            'Unit parity used another recognizer implementation')
    validate_commit(commit, sources)
    return {'schema': 1, 'scope': 'Actual packaged native-codec / wide-band verification on two spent generated development sources',
            'implementation_commit': commit, 'source_files_sha256': sources, 'protocol_sha256': digest(protocol_path),
            'independent_source_recordings': 2, 'fresh_efficacy_recordings': 0, 'new_tts_requests': 0,
            'http_attempts': 6, 'positive_whole_source_controls': 2, 'cases': cases,
            'recognizer_unit_parity': parity['cases'],
            'development_report_sha256': digest(development_path), 'quality_estimates_reused_only_for_identical_files': True,
            'interpretation': 'The tested implementation reproduces the two quality-qualified development selections and their exact-file repeated negative checks. Fresh validation remains pending.',
            'limits': ['Both sources were already used for development; repeated checks and packaging add no independent samples.',
                       'Turbo and MMS participate in selection. Their scores are not independent quality certification.',
                       'Supplemental Arabic HTTP role labels start at processed_2 because their counter included the opening original control. The actual verifier audit supplies the correct confirmation ordinals 1 and 2; both ledgers bind the same two requests and file hash.',
                       'Worker job timings exclude initial model loading and detector waits and cover two CPU examples only.',
                       'No human listening approval or physical watermark-erasure proof is claimed.'],
            'private_recordings_included': 0, 'recognized_words_exported': False, 'speaker_embeddings_exported': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--implementation-commit', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_runtime.json')
    args = parser.parse_args()
    report = build(args.directory, args.implementation_commit)
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError('Use a new report path for a different historical observation')
    args.output.write_text(text)
    print('Exported two spent exact-file runtime checks; zero fresh efficacy recordings')


if __name__ == '__main__':
    main()
