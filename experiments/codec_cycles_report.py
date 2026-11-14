"""Curate one spent long-file cycle example, retaining pending voice qualification."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export(directory):
    load = lambda path: json.loads(path.read_text())
    folder = directory/'codec_v2_development/long_cycle'
    generation = load(directory/'codec_cycles_long/result.json')
    assessment = load(folder/'assessment.json')
    metadata = load(folder/'metadata.json')
    endpoint = load(folder/'endpoint/result.json')
    summary = load(directory/'codec_v2_development/summary.json')
    if not all(v['complete'] for v in (generation, assessment, endpoint, summary)):
        raise ValueError('Incomplete long-file development evidence')
    source, candidate = assessment['source_sha256'], assessment['output_sha256']
    source_path = directory/'codec_fresh/originals'/(assessment['id']+'.flac')
    candidate_path = folder/'candidate.flac'
    if digest(source_path) != source or digest(candidate_path) != candidate:
        raise ValueError('A source or final whole file changed after its checks')
    first, second = sf.info(source_path), sf.info(candidate_path)
    if (first.frames, first.samplerate, first.channels) != (second.frames, second.samplerate, second.channels):
        raise ValueError('The complete sample grid changed')
    if (generation['source_sha256'] != source or endpoint['source_sha256'] != source or
            generation['cycles'][1]['sha256'] != candidate or endpoint['candidate_sha256'] != candidate or
            metadata['candidate_flac_sha256'] != candidate or not metadata['content_guard_passed'] or
            metadata['codec_cycle_passes'] != 2 or
            assessment['metadata_sha256'] != digest(folder/'metadata.json') or
            endpoint['assessment_sha256'] != digest(folder/'assessment.json')):
        raise ValueError('Generation, processing and detector inputs differ')
    for name, checksum in summary['source_files_sha256'].items():
        snapshot = subprocess.check_output(['git', 'show', assessment['implementation_commit']+':'+name], cwd=ROOT)
        if hashlib.sha256(snapshot).hexdigest() != checksum:
            raise ValueError('Measured processing source differs from its Git snapshot')
    expected = [('opening_original', True)]*2+[('candidate', False)]*2+[('closing_original', True)]*2
    if ([(r['role'], r['detected']) for r in endpoint['checks']] != expected or
            len(endpoint['http_attempts']) != 6 or any(r['http_status'] != 200 for r in endpoint['checks'])):
        raise ValueError('Retain the complete exact-file detector protocol')
    starts = [datetime.fromisoformat(r['started_at']) for r in endpoint['http_attempts']]
    if any((right-left).total_seconds() < 79.9 for left, right in zip(starts, starts[1:])):
        raise ValueError('Detector requests were not paced as declared')
    quality = load(directory/'codec_v2_development/quality.v2.json')
    if not quality['complete'] or not quality['originals_completed_before_candidate_grading']:
        raise ValueError('Selected-file ASR scoring is incomplete')
    asr = {}
    for name in ('small', 'turbo', 'mms', 'sensevoice'):
        metric = quality['metrics'][name]
        before = next(v for v in metric['originals'] if v['id'] == assessment['id'])
        after = next(v for v in metric['candidates'] if v['id'] == assessment['id'])
        if before['source_sha256'] != source or after['source_sha256'] != source or after['output_sha256'] != candidate:
            raise ValueError('ASR scores refer to different files')
        asr[name] = {'model': metric['model'], 'original': before, 'processed': after}
    gates = {name+'_reference_delta': asr[name]['processed']['reference_error_delta'] <= .03+1e-12 for name in ('small', 'turbo', 'mms')}
    gates.update(turbo_paired=asr['turbo']['processed']['paired_disagreement'] <= .03+1e-12,
                 mms_paired=asr['mms']['processed']['paired_disagreement'] <= .03+1e-12)
    voice = None
    voice_path = directory/'codec_profile_voice.json'
    if voice_path.exists():
        voice_report = load(voice_path)
        if voice_report['complete']:
            row = next(v for v in voice_report['clips'] if v['id'] == 'codec_v2_development__long_cycle')
            if row['source_sha256'] != source or row['output_sha256'] != candidate or row['status'] != 'scored':
                raise ValueError('Voice measurements refer to different complete files')
            voice = {'metric': voice_report['metric'], 'clip': row}
    seven = dict(gates, speaker_minimum=None if voice is None else voice['clip']['speaker_minimum'] >= .95-1e-12,
                 dnsmos_overall=None if voice is None else voice['clip']['dnsmos_delta']['overall'] >= -.2-1e-12)
    return {'kind': 'Spent development example: repeated native codec reconstruction',
        'fresh_independent_sources': 0, 'independent_source_recordings': 1,
        'frozen_primary_study_result_changed': False, 'human_listening_approval': False,
        'physical_erasure_proven': False, 'source_sha256': source, 'candidate_sha256': candidate,
        'source_format': {k: generation['cycles'][1][k] for k in ('samples', 'sample_rate', 'channels')},
        'generated_cycles': generation['cycles'], 'selected_cycle': 2,
        'processing': assessment, 'detector': endpoint, 'asr': asr,
        'five_asr_gate_results': gates, 'all_five_asr_gates_passed': all(gates.values()),
        'voice': voice, 'seven_quality_gate_results': seven,
        'voice_quality_scoring_complete': voice is not None,
        'all_seven_quality_gates_verified': voice is not None and all(seven.values()),
        'quality_status': ('All selected-file estimates are complete; inspect every gate.' if voice else
                           'Selected-file WavLM/DNSMOS scoring is queued; full qualification remains unproven.'),
        'limitations': ['One already-spent long source; cycles and requests are not new independent samples.',
            'The third generated cycle was not detector-tested.',
            'The frozen fresh-study result remains six fully qualified marked sources out of fourteen.',
            'A detector-negative result does not prove watermark erasure or universal reliability.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_cycles_development.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.directory), indent=2, allow_nan=False)+'\n')
    print(args.output)
