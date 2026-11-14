"""Export spent French development evidence without changing fresh-study results."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def curate(directory):
    read = lambda path: json.loads(path.read_text())
    repair = read(directory/'result.json')
    endpoint = read(directory/'endpoint/result.json')
    scores = {name: read(directory/(name+'.json')) for name in ('small', 'turbo', 'mms', 'voice')}
    source, candidate = repair['source_sha256'], repair['output_sha256']
    if (not repair['complete'] or not repair['content_guard_passed'] or
            not repair['rejected_file_reproduction_exact'] or repair['reference_text_used'] or
            not endpoint['complete'] or endpoint['candidate_detected'] or
            endpoint['source_sha256'] != source or endpoint['candidate_sha256'] != candidate or
            endpoint['assessment_sha256'] != digest(directory/'result.json')):
        raise ValueError('Incomplete or mismatched exact-file repair evidence')
    expected = [('opening_original', True)]*2+[('candidate', False)]*2+[('closing_original', True)]*2
    checks = endpoint['checks']
    if ([(v['role'], v['detected']) for v in checks] != expected or
            any(v['http_status'] != 200 for v in checks) or len(endpoint['http_attempts']) != 6):
        raise ValueError('The complete fixed detector protocol must be retained')
    rows = {}
    for name, report in scores.items():
        if not report['complete'] or len(report['clips']) != 1:
            raise ValueError('Missing selected-file scoring')
        row = report['clips'][0]
        if row['source_sha256'] != source or row['output_sha256'] != candidate or row['status'] != 'scored':
            raise ValueError('Quality measurements refer to different files')
        rows[name] = row
    gates = {name+'_reference_delta': rows[name]['reference_error_delta'] <= .03+1e-12 for name in ('small', 'turbo', 'mms')}
    gates.update(turbo_paired=rows['turbo']['paired_disagreement'] <= .03+1e-12,
                 mms_paired=rows['mms']['paired_disagreement'] <= .03+1e-12,
                 speaker_minimum=rows['voice']['speaker_minimum'] >= .95-1e-12,
                 dnsmos_overall=rows['voice']['dnsmos_delta']['overall'] >= -.2-1e-12)
    if not all(gates.values()):
        raise ValueError('This exporter describes the qualifying French development example')
    return {'kind': 'Spent development example; no independent fresh source',
        'fresh_independent_sources': 0, 'frozen_primary_study_result_changed': False,
        'source': {'id': rows['small']['id'], 'sha256': source}, 'candidate_sha256': candidate,
        'primary_failure': 'Native file stayed detected; band file exceeded the MMS consistency limit',
        'repair': repair, 'quality': {name: {'metric': scores[name]['metric'], 'clip': rows[name]} for name in scores},
        'quality_gate_results': gates, 'all_seven_estimated_quality_gates_passed': True,
        'detector': endpoint, 'evidence_sha256': {str(p.relative_to(directory)): digest(p) for p in
            [directory/'result.json', directory/'endpoint/result.json']+[directory/(n+'.json') for n in scores]},
        'limits': ['One previously failed source was used to develop this proposal.',
                   'The frozen fresh-study count remains six fully qualified marked sources out of fourteen.',
                   'Automated quality estimates do not establish human listening approval.',
                   'A detector-negative result is not proof of erasure or universal reliability.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability/codec_ctc_repair_fr')
    parser.add_argument('--output', type=Path, default=ROOT/'results/codec_repair_development.json')
    args = parser.parse_args()
    result = curate(args.directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(args.output)
