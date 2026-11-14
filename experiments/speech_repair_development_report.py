"""Export aggregate/metric-only evidence for spent speech-repair development.

No new held-out examples are claimed. Local transcripts and embeddings are
deliberately not copied. Source fingerprints below identify authored fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
IMPLEMENTATION_COMMIT = '05c244c'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def indexed(rows):
    result = {row['id']: row for row in rows}
    require(len(result) == len(rows), 'Duplicate metric IDs')
    return result


def build(directory):
    plan_path = directory/'word_repair_vocoder_dev.plan.json'
    plan = load(plan_path)
    protocol_path = directory/'word_repair_vocoder_dev.api.plan.json'
    protocol = load(protocol_path)
    api = load(directory/'word_repair_vocoder_dev.api.json')
    require(api['measurement_complete'] is True and api['protocol_sha256'] == digest(protocol_path),
            'Detector evidence is incomplete or uses another protocol')
    require(protocol['plan_sha256'] == digest(plan_path) and protocol['required_negative_checks'] == 2,
            'Wrong development plan or confirmation count')
    metrics = {}
    for suffix in ('asr', 'asr_turbo', 'voice_quality'):
        path = directory/('word_repair_vocoder_dev.'+suffix+'.json')
        document = load(path)
        require(document['complete'] is True and document['plan_sha256'] == digest(plan_path),
                'Incomplete quality measurement')
        require(protocol['quality_reports_sha256'][suffix] == digest(path), 'Quality evidence changed')
        metrics[suffix] = indexed(document['clips'])
    mms = load(directory/'mms_review_development.json')
    require(mms['complete'] is True and mms['originals_scored_before_any_candidate'] is True,
            'Independent review is incomplete')
    packaged = load(directory/'speech_repair_package_smoke/result.json')
    package_rows = indexed(packaged['cases'])
    implementation = subprocess.check_output(
        ['git', 'rev-parse', IMPLEMENTATION_COMMIT], cwd=ROOT, text=True).strip()
    for name, expected in packaged['source_sha256'].items():
        frozen = subprocess.check_output(['git', 'show', implementation+':tools/'+name], cwd=ROOT)
        require(hashlib.sha256(frozen).hexdigest() == expected, 'Packaged source is not in the cited commit')
    cases = []
    for item in protocol['selected']:
        key, base_key = item['id'], item['base_id']
        source, output = ROOT/item['source'], ROOT/item['output']
        source_sha, output_sha = digest(source), digest(output)
        require(source_sha == item['sha256'], 'Development source changed')
        observations = [r for r in api['clips'] if r['id'] == key]
        require(sorted(r['role'] for r in observations) == ['processed', 'processed_repeat'] and
                all(r['http_status'] == 200 and r['sha256'] == output_sha and r['detected'] is False
                    for r in observations), 'Missing exact-file negative checks')
        originals = [r for r in api['clips'] if r['id'] == base_key and r['role'] == 'original']
        require(len(originals) == 1 and originals[0]['sha256'] == source_sha and
                originals[0]['http_status'] == 200 and originals[0]['detected'] is True,
                'Missing original positive')
        estimated = {}
        for suffix in ('asr', 'asr_turbo'):
            metric = metrics[suffix][key]
            require(metric['source_sha256'] == source_sha and metric['output_sha256'] == output_sha,
                    'Quality measured different bytes')
            estimated[suffix] = {k: metric[k] for k in
                                ('asr_disagreement', 'original_reference_error', 'processed_reference_error',
                                 'reference_error_delta', 'reference_units')}
        voice = metrics['voice_quality'][key]
        require(voice['source_sha256'] == source_sha and voice['output_sha256'] == output_sha,
                'Voice estimates measured different bytes')
        estimated['voice'] = {k: voice[k] for k in ('speaker_minimum', 'speaker_mean', 'dnsmos_delta')}
        packaged_path = directory/'speech_repair_package_smoke'/(base_key+'.wav')
        pcm, rate = sf.read(output, dtype='float32')
        packaged_pcm, packaged_rate = sf.read(packaged_path, dtype='float32')
        require(rate == packaged_rate and np.array_equal(pcm, packaged_pcm) and
                package_rows[base_key]['decoded_pcm_exact'] is True, 'Packaging parity failed')
        independent = [r for r in mms['clips'] if r['id'] == base_key]
        before = next(r for r in independent if r['role'] == 'original')
        after = next(r for r in independent if r['role'] == 'word_repaired')
        require(before['source_sha256'] == source_sha and after['output_sha256'] == digest(packaged_path)
                and before['calibration_eligible'] is True, 'Independent review measured another source')
        estimated['mms'] = {'original_reference_error': before['reference_error'],
                            **{k: after[k] for k in ('reference_error', 'reference_error_delta', 'paired_disagreement')}}
        cases.append({'source_id': base_key, 'language': item['language'], 'seconds': item['seconds'],
                      'source_sha256': source_sha, 'output_sha256': output_sha,
                      'original_detected': True, 'processed_detected': [r['detected'] for r in observations],
                      'quality_estimates': estimated, 'packaging': package_rows[base_key]})
    controls = [r for r in api['clips'] if r['role'].endswith('positive_control')]
    require(len(controls) == 2 and all(r['detected'] is True and r['http_status'] == 200 for r in controls),
            'Service controls failed')
    cli = load(directory/'speech_repair_cli_smoke.complete.json')
    require(cli['status'] == 'passed' and cli['parent_without_torch'] is True and
            cli['positive_controls'] == 2 and cli['negative_checks'] == 2 and
            cli['audit_sha256'] == digest(directory/'speech_repair_cli_smoke.audit.json') and
            cli['output_sha256'] == digest(directory/'speech_repair_cli_smoke.flac') and
            cli['output_sha256'] == cases[0]['output_sha256'], 'CLI integration failed')
    runtime = load(directory/'regional_vocoder_probe.json')
    require(len(runtime['cases']) == 2 and all(r['regional_used_pcm_exact_to_full'] is True and
            r['patched_flac_exact_to_development_candidate'] is True for r in runtime['cases']),
            'Regional equivalence failed')
    union = load(directory/'ensemble_region_dev/result.json')
    require(union['source_sha256'] == cases[0]['source_sha256'] and
            union['output_sha256'] == digest(directory/'ensemble_region_dev/arabic_union.flac'),
            'Union-region development probe changed')
    return {'schema': 1, 'implementation_commit': implementation,
            'scope': 'Two already-spent authored development sources; zero new held-out recordings',
            'fresh_recordings': 0, 'development_recordings': len(cases),
            'status': 'Independent Arabic content regression unresolved; not a generally validated remover',
            'frozen_studies_unchanged': True, 'private_recordings_included': 0,
            'primary_selection_before_repair_detector_labels': protocol['selection_rule'],
            'independent_review': {k: mms[k] for k in
                                   ('model', 'revision', 'license', 'original_calibration_maximum_reference_error',
                                    'normalization', 'originals_scored_before_any_candidate')},
            'cases': cases, 'detector_requests': len(api['http_attempts']),
            'positive_service_controls': len(controls), 'actual_cli_integration': cli,
            'regional_donor_runtime': runtime,
            'posthoc_union_region_probe': {
                'scope': 'Same spent Arabic source; union of Turbo and MMS disputed regions',
                **{k: union[k] for k in ('source_sha256', 'output_sha256', 'coverage',
                                        'mms_paired_disagreement', 'mms_reference_error_delta')},
                'mms_reference_delta_threshold': .03, 'mms_reference_delta_passed': False,
                'detector_checked': False, 'runtime_integration': False},
            'limits': ['Recognizer estimates and detector-negative responses cannot prove physical erasure or unchanged words.',
                       'Turbo is both the runtime guard and one evaluator; MMS is a separate architecture.',
                       'Arabic MMS reference error increased by 6.45 percentage points after repair.',
                       'No human listening approval was obtained.',
                       'Timing observations cover two recordings on one machine; the complete guarded path is slower than base reconstruction.'],
            'recognized_transcripts_exported': False, 'speaker_embeddings_exported': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/speech_repair_development.json')
    args = parser.parse_args()
    report = build(args.directory)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Exported', len(report['cases']), 'spent development cases; Arabic failure retained')


if __name__ == '__main__':
    main()
