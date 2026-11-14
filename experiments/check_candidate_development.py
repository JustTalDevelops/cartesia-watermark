"""Check one locally assessed spent candidate with exact-file controls.

Six requests maximum: opening marked source twice, candidate twice, closing
marked source twice. Does not publish audio or add independent fresh samples.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'tools'))
os.environ['DETECT_PROXY'] = ''
os.environ['DETECT_INTERVAL'] = '80'
import cartesia


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def check(source, candidate, assessment_path, directory):
    assessment = json.loads(assessment_path.read_text())
    source_sha, candidate_sha = digest(source), digest(candidate)
    if (assessment['source_sha256'] != source_sha or assessment['output_sha256'] != candidate_sha or
            assessment['content_guard_passed'] is not True or not assessment['complete']):
        raise ValueError('Use the exact complete locally assessed candidate and source')
    first, second = sf.info(source), sf.info(candidate)
    if ((first.samplerate, first.frames, first.channels) != (second.samplerate, second.frames, second.channels) or
            not 5 <= first.duration <= 300 or max(source.stat().st_size, candidate.stat().st_size) > 32_000_000):
        raise ValueError('Use matching complete whole files within the detector limits')
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'result.json'
    report = {'source_sha256': source_sha, 'candidate_sha256': candidate_sha,
              'assessment_sha256': digest(assessment_path), 'checker_source_sha256': digest(Path(__file__)),
              'minimum_interval_seconds': 80, 'maximum_http_attempts': 6,
              'fresh_independent_sources': 0, 'frozen_study_result_changed': False,
              'human_quality_verified': False, 'http_attempts': [], 'checks': [], 'complete': False}
    current = {}

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(path)

    save()
    original_post = cartesia.requests.post

    def counted(*args, **kwargs):
        if len(report['http_attempts']) >= 6:
            raise RuntimeError('The development request ceiling has been reached')
        row = dict(current, started_at=now())
        report['http_attempts'].append(row)
        save()
        try:
            response = original_post(*args, **kwargs)
        except Exception as error:
            row['error_type'] = type(error).__name__
            save()
            # The legacy client retries ConnectionError. This fixed protocol
            # stops instead, retaining the uncertain HTTP attempt.
            raise RuntimeError('Detector connection failed; no automatic retry') from error
        row['http_status'] = response.status_code
        save()
        return response

    cartesia.requests.post = counted
    try:
        for role, target, expected, checksum in (
                ('opening_original', source, True, source_sha),
                ('candidate', candidate, None, candidate_sha),
                ('closing_original', source, True, source_sha)):
            results = []
            for ordinal in (1, 2):
                if digest(source) != source_sha or digest(candidate) != candidate_sha:
                    raise ValueError('Source or candidate changed during detector checks')
                current.clear()
                current.update(role=role, ordinal=ordinal, sha256=checksum)
                status, response = cartesia.detect(target)
                if status != 200 or not isinstance(response, dict) or type(response.get('detected')) is not bool:
                    raise RuntimeError('Detector unavailable; no repeated request')
                if digest(target) != checksum:
                    raise ValueError('Exact detector input changed')
                report['checks'].append(dict(current, detected=response['detected'], http_status=status, checked_at=now()))
                results.append(response['detected'])
                save()
                print(role, ordinal, response['detected'], flush=True)
            if results[0] != results[1] or (expected is not None and results[0] != expected):
                raise RuntimeError('Inconsistent detector responses or failed original control')
            if role == 'candidate':
                report['candidate_detected'] = results[0]
        report.update(complete=True, status='checked_positive' if report['candidate_detected'] else 'checked_negative', completed_at=now())
        save()
    except Exception as error:
        report.update(stopped_at=now(), error_type=type(error).__name__)
        save()
        raise
    finally:
        cartesia.requests.post = original_post
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    check(args.source, args.candidate, args.assessment, args.directory)
