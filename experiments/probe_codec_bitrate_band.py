"""Assess fixed 3/6 kbps band reconstructions on a spent codec counterexample.

Uses existing exact native-codec files and the pinned band worker. No target
text, detector requests or verified publication; every candidate remains a
spent development treatment of the same complete recording.
"""
import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def probe(source, bitrate_directory, directory, guard_tools):
    sys.path.insert(0, str(guard_tools))
    from codec_audio import digest, quantize, validate_audio
    from codec_content import ContentRecognizer, assess
    from reconstruction_worker import ReconstructionWorker
    provenance = json.loads((bitrate_directory/'result.json').read_text())
    source_sha = digest(source)
    if provenance['source_sha256'] != source_sha or not provenance['complete']:
        raise ValueError('Require complete exact spent bitrate-generation evidence')
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'result.json'
    report = {'source_sha256': source_sha, 'bitrate_report_sha256': digest(bitrate_directory/'result.json'),
        'probe_source_sha256': digest(Path(__file__)), 'fresh_independent_sources': 0,
        'reference_text_used': False, 'cartesia_requests': 0, 'quality_verified': False,
        'frozen_primary_result_changed': False, 'candidates': [], 'complete': False}

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

    save()
    audio, rate = sf.read(source, dtype='float32')
    validate_audio(audio, rate)
    recognizer = ContentRecognizer()
    before = recognizer.transcribe(source)
    worker = ReconstructionWorker(ROOT/'.venv/bin/python', ROOT/'tools/reconstruct_codec_band.py')
    try:
        for bandwidth in (3., 6.):
            generation = next(v for v in provenance['candidates'] if v['bandwidth_kbps'] == bandwidth)
            native = bitrate_directory/('codec_'+str(bandwidth).replace('.', '_')+'kbps.flac')
            if digest(native) != generation['output_sha256']:
                raise ValueError('Existing codec file differs from its original assessment')
            folder = directory/str(int(bandwidth))
            folder.mkdir()
            manifest = folder/'input.json'
            manifest.write_text(json.dumps({'source': str(source.resolve()), 'source_sha256': source_sha,
                'native': str(native.resolve()), 'native_sha256': digest(native)}))
            output = folder/'candidate.wav'
            info = worker.run(manifest, output, 'codec-band', folder/'band.json', timeout=max(300, len(audio)/rate*8+120))
            candidate_audio, candidate_rate = sf.read(output, dtype='float32')
            if (candidate_rate != rate or candidate_audio.shape != audio.shape or
                    info['source_sha256'] != source_sha or info['native_sha256'] != digest(native) or
                    info['output_sha256'] != digest(output)):
                raise ValueError('Band reconstruction changed its complete source or sample grid')
            candidate = folder/'candidate.flac'
            quantize(candidate, candidate_audio, rate)
            checksum = digest(candidate)
            if checksum != info['encoded_flac_sha256']:
                raise ValueError('Final PCM24 encoding differs from the worker result')
            after = recognizer.transcribe(candidate, before['language'])
            if digest(source) != source_sha or digest(candidate) != checksum:
                raise ValueError('A source or candidate changed during the complete-file guard')
            row = {'source_sha256': source_sha, 'output_sha256': checksum, 'native_bandwidth_kbps': bandwidth,
                'native_sha256': digest(native), 'band_metadata_sha256': digest(folder/'band.json'),
                'complete': True, 'quality_verified': False, **assess(before, after)}
            (folder/'assessment.json').write_text(json.dumps(row, indent=2)+'\n')
            report['candidates'].append(row)
            save()
            print('BAND_ASSESSED', json.dumps(row), flush=True)
        report['complete'] = True
        save()
    finally:
        worker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bitrate-directory', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--guard-tools', type=Path, required=True)
    args = parser.parse_args()
    probe(args.source, args.bitrate_directory, args.directory, args.guard_tools)
