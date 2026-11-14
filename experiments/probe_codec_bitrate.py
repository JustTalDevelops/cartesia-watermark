"""Assess fixed lower bitrates on a spent codec counterexample, without API calls.

The pinned NativeCodec DSP and weights are reused. Only the checkpoint's
supported bandwidth ID changes; quality references never enter this process.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def probe(source, audit_path, directory, guard_tools, bandwidths):
    sys.path.insert(0, str(guard_tools))
    from codec_audio import digest, quantize, validate_audio
    from codec_content import ContentRecognizer, assess
    from codec_inference import NativeCodec
    import torch
    audit = json.loads(audit_path.read_text())
    source_sha = digest(source)
    if source_sha != audit['source_sha256'] or audit['status'] != 'not_cleared':
        raise ValueError('Use an exact spent unsuccessful whole-file source')
    directory.mkdir(parents=True, exist_ok=False)
    report = {'source_sha256': source_sha, 'audit_sha256': digest(audit_path),
        'probe_source_sha256': digest(Path(__file__)), 'fresh_independent_sources': 0,
        'reference_text_used': False, 'cartesia_requests': 0, 'quality_verified': False,
        'guard_source_sha256': {p.name: digest(p) for p in guard_tools.glob('codec_*.py')},
        'candidates': [], 'complete': False}
    path = directory/'result.json'

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

    save()
    audio, rate = sf.read(source, dtype='float32')
    validate_audio(audio, rate)
    codec, recognizer = NativeCodec(), ContentRecognizer()
    before = recognizer.transcribe(source)
    if not before['supported']:
        raise ValueError('Source language is unsupported by the selected guards')
    pinned_model = codec.model
    ids = {1.5: 0, 3.: 1, 6.: 2, 12.: 3}
    for bandwidth in bandwidths:
        started = time.perf_counter()

        class AtBandwidth:
            def __call__(self, tensor, **unused):
                return pinned_model(tensor, bandwidth_id=torch.tensor([ids[bandwidth]]))

        codec.model = AtBandwidth()
        rebuilt, gain = codec.reconstruct(audio, rate)
        candidate = directory/('codec_'+str(bandwidth).replace('.', '_')+'kbps.flac')
        quantize(candidate, rebuilt, rate)
        checksum = digest(candidate)
        after = recognizer.transcribe(candidate, before['language'])
        if digest(source) != source_sha or digest(candidate) != checksum:
            raise ValueError('A complete source or candidate changed during assessment')
        row = {'source_sha256': source_sha, 'output_sha256': checksum, 'bandwidth_kbps': bandwidth,
            'gain': gain, 'samples': len(audio), 'sample_rate': rate,
            'channels': 1 if audio.ndim == 1 else audio.shape[1], 'quality_verified': False,
            'complete': True, 'elapsed_seconds': time.perf_counter()-started, **assess(before, after)}
        report['candidates'].append(row)
        (directory/(candidate.stem+'.assessment.json')).write_text(json.dumps(row, indent=2)+'\n')
        save()
        print('BITRATE_ASSESSED', json.dumps(row), flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--guard-tools', type=Path, default=ROOT/'tools')
    parser.add_argument('--bandwidths', type=float, choices=(1.5, 3., 6., 12.), nargs='+', default=[6., 3.])
    args = parser.parse_args()
    probe(args.source, args.audit, args.directory, args.guard_tools, args.bandwidths)
