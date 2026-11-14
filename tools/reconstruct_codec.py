"""Experimental codec reconstruction guarded by two local recognizers.

Processing is local. The separate verification CLI uploads whole candidates to
Cartesia. Neither recognizer agreement nor a negative detector proves erasure.
"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import select
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

from codec_audio import digest, quantize, validate_audio, validate_paths
from codec_content import ContentRecognizer, assess
from codec_inference import NativeCodec, ROOT
from codec_settings import PRESETS, SETTINGS
from reconstruction_worker import ReconstructionWorker
from speech_repair import pcm_digest


class CodecRuntime:
    def __init__(self, codec=None, recognizer=None, worker_factory=ReconstructionWorker):
        self.codec = codec if codec is not None else NativeCodec()
        self.recognizer = recognizer if recognizer is not None else ContentRecognizer()
        self.worker_factory, self.band_worker = worker_factory, None
        self.cache, self.temporary = None, None

    def clear_job(self):
        self.cache = None
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None

    def close(self):
        self.clear_job()
        if self.band_worker is not None:
            self.band_worker.close()
            self.band_worker = None

    def job(self, source, original_sha, audio, rate, parent):
        key = (str(source.resolve()), original_sha, str(parent.resolve()))
        if self.cache is not None and self.cache['key'] == key and self.cache['native'].is_file():
            return self.cache, True
        self.clear_job()
        self.temporary = tempfile.TemporaryDirectory(prefix='.codec-source-', dir=parent)
        directory = Path(self.temporary.name)
        before = self.recognizer.transcribe(source)
        job = {'key': key, 'directory': directory, 'before': before, 'native': directory/'native.flac'}
        if before['supported'] and before['turbo'] and before['mms']:
            started = time.monotonic()
            rebuilt, gain = self.codec.reconstruct(audio, rate)
            quantize(job['native'], rebuilt, rate)
            job.update(native_gain=gain, native_elapsed_seconds=time.monotonic()-started,
                       native_sha256=digest(job['native']))
        self.cache = job
        return job, False

    def process_file(self, source, output, preset='codec-native', metadata=None, retain_job=False):
        validate_paths(source, output, metadata)
        if preset not in PRESETS:
            raise ValueError('Unknown codec preset')
        started = time.monotonic()
        original_sha = digest(source)
        audio, rate = sf.read(source, dtype='float32')
        validate_audio(audio, rate)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {'preset': preset, 'settings': SETTINGS, 'source_sha256': original_sha,
                  'sample_rate': rate, 'samples': len(audio), 'channels': 1 if audio.ndim == 1 else audio.shape[1],
                  'codec_model': self.codec.metadata, 'recognizers': self.recognizer.metadata,
                  'content_guard_passed': False, 'quality_verified': False, 'output_sha256': None,
                  'meaning': 'Two recognizer-consistency estimates; no human approval or unchanged-speech/erasure guarantee'}
        output_identity = None
        try:
            job, reused = self.job(source, original_sha, audio, rate, output.parent)
            report.update(source_job_reused=reused, language=job['before']['language'])
            if not job['before']['supported']:
                report['rejection_reason'] = 'unsupported_language_for_content_guard'
            elif not job['before']['turbo'] or not job['before']['mms']:
                report['rejection_reason'] = 'original_speech_not_recognized'
            else:
                candidate = job['native']
                if digest(candidate) != job['native_sha256']:
                    raise RuntimeError('Cached native codec recording changed')
                report.update(native_codec_sha256=job['native_sha256'], native_gain=job['native_gain'],
                              native_elapsed_seconds=job['native_elapsed_seconds'])
                with tempfile.TemporaryDirectory(prefix='.codec-assessment-', dir=output.parent) as temporary:
                    directory = Path(temporary)
                    if preset == 'codec-band':
                        manifest = directory/'band.input.json'
                        manifest.write_text(json.dumps({'source': str(source.resolve()), 'source_sha256': original_sha,
                                                        'native': str(candidate), 'native_sha256': job['native_sha256']}))
                        if self.band_worker is None:
                            self.band_worker = self.worker_factory(ROOT/'.venv/bin/python', ROOT/'tools/reconstruct_codec_band.py',
                                                                   isolated=False)
                        band_path = directory/'band.wav'
                        band_info = self.band_worker.run(manifest, band_path, preset, directory/'band.json',
                                                         timeout=max(300, len(audio)/rate*8+120))
                        mixed, band_rate = sf.read(band_path, dtype='float32')
                        if (band_info.get('output_sha256') != digest(band_path) or band_rate != rate or mixed.shape != audio.shape or
                                not np.isfinite(mixed).all() or
                                band_info.get('source_sha256') != original_sha or band_info.get('native_sha256') != job['native_sha256']):
                            raise RuntimeError('Band worker returned a mismatched recording')
                        candidate = directory/'band.flac'
                        quantize(candidate, mixed, rate)
                        if digest(candidate) != band_info.get('encoded_flac_sha256'):
                            raise RuntimeError('Band worker publication changed its encoded recording')
                        report['band_model'] = band_info
                    candidate_sha = digest(candidate)
                    result, candidate_rate = sf.read(candidate, dtype='float32')
                    if candidate_rate != rate or result.shape != audio.shape or not np.isfinite(result).all() or np.max(np.abs(result)) >= 1:
                        raise RuntimeError('Candidate changed the format or returned invalid samples')
                    after = self.recognizer.transcribe(candidate, job['before']['language'])
                    if digest(candidate) != candidate_sha or digest(source) != original_sha:
                        raise RuntimeError('A source or candidate changed during content assessment')
                    report.update(assess(job['before'], after), candidate_flac_sha256=candidate_sha)
                    if report['content_guard_passed']:
                        report.update(content_guard_pcm_sha256=pcm_digest(result), content_guard_sample_grid='PCM_24')
                        with output.open('xb') as handle:
                            stat = os.fstat(handle.fileno())
                            output_identity = (stat.st_dev, stat.st_ino)
                            sf.write(handle, result, rate, format='WAV', subtype='FLOAT')
                        report['output_sha256'] = digest(output)
                    else:
                        report['rejection_reason'] = 'recognizer_disagreement'
            if digest(source) != original_sha:
                raise RuntimeError('Source changed during codec processing')
            report['elapsed_seconds'] = time.monotonic()-started
            if metadata:
                metadata.parent.mkdir(parents=True, exist_ok=True)
                with metadata.open('x') as handle:
                    json.dump(report, handle, indent=2)
            return report
        except BaseException:
            if output_identity is not None:
                try:
                    stat = output.stat()
                    if (stat.st_dev, stat.st_ino) == output_identity:
                        output.unlink()
                except FileNotFoundError:
                    pass
            self.clear_job()
            raise
        finally:
            if not retain_job:
                self.clear_job()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--preset', choices=PRESETS, default='codec-native')
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.input or args.output or args.metadata or args.check:
            parser.error('Worker mode does not accept file arguments')
    elif not args.check:
        if not args.input or not args.output:
            parser.error('Input and output are required')
        validate_paths(args.input, args.output, args.metadata)
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        runtime = CodecRuntime()
        try:
            if args.worker:
                print(json.dumps({'ready': True}), file=protocol, flush=True)
                while True:
                    # Source audio/recognized units are cached only across an
                    # immediate content-rejected fallback, then cleared idle.
                    if not select.select([sys.stdin], [], [], 10)[0]:
                        runtime.clear_job()
                        continue
                    line = sys.stdin.readline()
                    if not line:
                        break
                    try:
                        request = json.loads(line)
                        report = runtime.process_file(Path(request['input']), Path(request['output']), request['preset'],
                                                      Path(request['metadata']) if request.get('metadata') else None, retain_job=True)
                        reply = {'report': report}
                    except Exception as error:
                        reply = {'error': f'{type(error).__name__}: {error}'}
                    print(json.dumps(reply), file=protocol, flush=True)
                return 0
            report = {'ready': True, 'presets': list(PRESETS), 'recognizers': runtime.recognizer.metadata} if args.check else runtime.process_file(
                args.input, args.output, args.preset, args.metadata)
        finally:
            runtime.close()
    print(json.dumps(report), flush=True)
    return 0 if report.get('content_guard_passed', True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
