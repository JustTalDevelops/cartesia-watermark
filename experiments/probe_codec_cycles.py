"""Generate repeated native codec passes for a spent detector-positive source.

Exact first-pass FLAC reproduction is required. Later passes are unassessed
development candidates, with no Cartesia requests or source reference text.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest, quantize, validate_audio
from codec_inference import NativeCodec


def probe(source, audit_path, directory, iterations=3):
    audit = json.loads(audit_path.read_text())
    source_sha = digest(source)
    native = next(a for a in audit['attempts'] if a['preset'] == 'codec-native')
    if (audit['source_sha256'] != source_sha or audit['status'] != 'not_cleared' or
            native.get('detected') is not True):
        raise ValueError('Use an exact spent source whose complete native file remained detected')
    directory.mkdir(parents=True, exist_ok=False)
    report = {'source_sha256': source_sha, 'audit_sha256': digest(audit_path),
              'probe_source_sha256': digest(Path(__file__)), 'fresh_independent_sources': 0,
              'reference_text_used': False, 'cartesia_requests': 0, 'quality_verified': False,
              'content_guard_passed': False, 'frozen_study_result_changed': False,
              'cycles': [], 'complete': False}
    path = directory/'result.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    audio, rate = sf.read(source, dtype='float32')
    validate_audio(audio, rate)
    shape = audio.shape
    codec = NativeCodec()
    for number in range(1, iterations+1):
        started = time.perf_counter()
        rebuilt, gain = codec.reconstruct(audio, rate)
        target = directory/f'cycle_{number:02d}.flac'
        audio = quantize(target, rebuilt, rate)
        checksum = digest(target)
        if audio.shape != shape:
            raise ValueError('A repeated pass changed the complete sample grid')
        if number == 1 and checksum != native['model']['native_codec_sha256']:
            raise ValueError('First pass differs from the exact measured native file')
        report['cycles'].append({'cycle': number, 'sha256': checksum, 'gain': gain,
                                 'elapsed_seconds': time.perf_counter()-started,
                                 'samples': len(audio), 'sample_rate': rate,
                                 'channels': 1 if audio.ndim == 1 else audio.shape[1]})
        path.write_text(json.dumps(report, indent=2)+'\n')
        print('CYCLE_GENERATED', number, checksum, flush=True)
    if digest(source) != source_sha:
        raise ValueError('Source changed during the development probe')
    report['complete'] = True
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--iterations', type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    probe(args.source, args.audit, args.directory, args.iterations)
