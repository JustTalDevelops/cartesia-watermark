"""Curate the spent Korean lower-bitrate counterexample, including every failure."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export(directory):
    load = lambda path: json.loads(path.read_text())
    native = directory/'codec_bitrate_ko_v2'
    band = directory/'codec_bitrate_ko_band'
    generation, reconstruction = load(native/'result.json'), load(band/'result.json')
    quality = load(band/'quality.json')
    if not all(v['complete'] for v in (generation, reconstruction, quality)):
        raise ValueError('Every planned bitrate and selected-file quality measurement must be complete')
    source = generation['source_sha256']
    if reconstruction['source_sha256'] != source or quality['source_sha256'] != source:
        raise ValueError('Development reports refer to different sources')
    source_path = directory/'codec_fresh/originals/codec_fresh_07_ko_sonic-preview.flac'
    if digest(source_path) != source:
        raise ValueError('Original whole-file source changed')
    rows = []
    for mode, bitrate, folder, candidate_path, assessment_path, endpoint_folder in (
            ('native', 3, native, native/'codec_3_0kbps.flac', native/'codec_3_0kbps.assessment.json', None),
            ('native', 6, native, native/'codec_6_0kbps.flac', native/'codec_6_0kbps.assessment.json', native/'endpoint6'),
            ('band', 3, band/'3', band/'3/candidate.flac', band/'3/assessment.json', band/'3/endpoint'),
            ('band', 6, band/'6', band/'6/candidate.flac', band/'6/assessment.json', band/'6/endpoint')):
        assessment = load(assessment_path)
        checksum = digest(candidate_path)
        if assessment['source_sha256'] != source or assessment['output_sha256'] != checksum or not assessment['complete']:
            raise ValueError('An assessed complete candidate changed')
        row = {'mode': mode, 'native_codec_bandwidth_kbps': bitrate, 'candidate_sha256': checksum,
               'content_guard_passed': assessment['content_guard_passed'],
               'paired_disagreement': assessment['paired_disagreement'], 'quality_gate_results': None,
               'detector': None, 'fresh_independent_sources': 0}
        if endpoint_folder is None:
            if assessment['content_guard_passed']:
                raise ValueError('The unqueried native 3 kbps file must retain its speech rejection')
            row['status'] = 'speech_rejected_not_detector_tested'
        else:
            endpoint = load(endpoint_folder/'result.json')
            expected = [('opening_original', True)]*2+[('candidate', True)]*2+[('closing_original', True)]*2
            if (not endpoint['complete'] or endpoint['candidate_detected'] is not True or
                    endpoint['source_sha256'] != source or endpoint['candidate_sha256'] != checksum or
                    endpoint['assessment_sha256'] != digest(assessment_path) or
                    [(c['role'], c['detected']) for c in endpoint['checks']] != expected or
                    len(endpoint['http_attempts']) != 6 or any(c['http_status'] != 200 for c in endpoint['checks'])):
                raise ValueError('Retain every repeated positive response and source control')
            starts = [datetime.fromisoformat(c['started_at']) for c in endpoint['http_attempts']]
            if any((b-a).total_seconds() < 79.9 for a, b in zip(starts, starts[1:])):
                raise ValueError('Detector pacing differs from the fixed protocol')
            row.update(detector=endpoint, status='still_detected_twice')
        if mode == 'band':
            for name, metric in quality['metrics'].items():
                measured = next(v for v in metric['candidates'] if v['native_bandwidth_kbps'] == bitrate)
                if measured['output_sha256'] != checksum:
                    raise ValueError('Quality estimates refer to a different band recording')
            voice = next(v for v in quality['voice']['clips'] if v['id'] == 'ko_band'+str(bitrate))
            if voice['source_sha256'] != source or voice['output_sha256'] != checksum:
                raise ValueError('Voice estimates refer to another complete file')
            row['quality_gate_results'] = quality['quality_gate_results'][str(bitrate)]
            row['all_seven_estimated_quality_gates_passed'] = all(row['quality_gate_results'].values())
        rows.append(row)
    return {'kind': 'Spent lower-bitrate development counterexample', 'source_sha256': source,
        'independent_source_recordings': 1, 'fresh_independent_sources': 0,
        'frozen_primary_study_result_changed': False, 'new_tts_requests': 0, 'candidates': rows,
        'quality': quality, 'successful_removals': 0, 'human_listening_approval': False,
        'physical_erasure_proven': False,
        'limits': ['Different bitrates and repeated queries are treatments of one already-used recording.',
            'Both band files passed seven estimated quality gates but remained detected twice.',
            'The speech-rejected native 3 kbps file was not detector-tested.',
            'MMS is retained as a Korean diagnostic; its original calibration is poor. The prospective guard uses SenseVoice.',
            'These observations do not change the frozen fresh-study result of six qualified marked sources out of fourteen.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_bitrate_development.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.directory), indent=2, allow_nan=False)+'\n')
    print(args.output)
