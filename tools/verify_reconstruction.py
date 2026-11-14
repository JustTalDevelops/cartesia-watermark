"""Reconstruct speech and publish only exact files checked negative by Cartesia.

This command uploads candidates to Cartesia's detector. It verifies the current
detector response, not physical watermark erasure or voice/content fidelity.
The regular web app continues to process recordings without external requests.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

import numpy as np
import soundfile as sf

from reconstruction_worker import ReconstructionWorker
from speech_repair import PRESETS as SPEECH_PRESETS, pcm_digest
from codec_settings import PRESETS as CODEC_PRESETS

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ('standard', 'full-flow', 'strong', 'smoothed-vocoder', 'filtered-prior')+tuple(SPEECH_PRESETS)+CODEC_PRESETS
GUARDED_PRESETS = tuple(SPEECH_PRESETS)+CODEC_PRESETS
MAX_BYTES = 32_000_000


class NotCleared(RuntimeError):
    """No acceptable candidate passed the requested checks."""


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_report(path, report, create=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if create:
        with path.open('x') as handle:
            json.dump(report, handle, indent=2)
        return
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_own_output(output, identity):
    """Preserve a file another process replaced after publication."""
    try:
        current = output.stat()
        if (current.st_dev, current.st_ino) == identity:
            output.unlink()
    except FileNotFoundError:
        pass


def publish_exact(source, output, expected):
    temporary, published_identity = None, None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=output.parent, prefix='.'+output.name+'.',
                                         suffix='.tmp', delete=False) as target:
            temporary = Path(target.name)
            with source.open('rb') as handle:
                shutil.copyfileobj(handle, target)
        if digest(temporary) != expected:
            raise RuntimeError('Copied bytes differ from the checked file')
        identity = temporary.stat()
        # A same-filesystem hard link publishes a complete file atomically and
        # refuses to replace an output created while verification was running.
        os.link(temporary, output)
        published_identity = (identity.st_dev, identity.st_ino)
        if digest(output) != expected:
            raise RuntimeError('Published bytes differ from the checked file')
        return published_identity
    except BaseException:
        if published_identity is not None:
            remove_own_output(output, published_identity)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reconstruct_verified(source, output, report_path, presets=('full-flow', 'strong'),
                         worker=None, detector=None, boundary_safe=False, confirmations=1):
    """Return an audit record only after publishing the exact checked bytes.

    On a positive result, try the next fixed preset. API errors, unavailable
    verification, changed files, and exhausted presets never publish an output.
    The caller owns a supplied worker; otherwise this function closes its worker.
    """
    source, output, report_path = map(Path, (source, output, report_path))
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Choose a new output file separate from the input')
    if report_path.exists() or report_path.resolve() in (source.resolve(), output.resolve()):
        raise ValueError('Choose a new report path separate from the audio files')
    if output.suffix.lower() not in ('.wav', '.flac'):
        raise ValueError('Output must be WAV or FLAC')
    if not presets or any(preset not in PRESETS for preset in presets):
        raise ValueError('Choose at least one supported reconstruction preset')
    if type(confirmations) is not int or not 1 <= confirmations <= 3:
        raise ValueError('Choose between one and three negative confirmations')
    audio, sample_rate = sf.read(source)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    if channels not in (1, 2) or not np.isfinite(audio).all() or not 8000 <= sample_rate <= 96000:
        raise ValueError('Choose finite mono/stereo audio at 8–96 kHz')
    seconds = len(audio)/sample_rate
    if not 5 <= seconds <= 300:
        raise ValueError('Whole-file verification requires 5 seconds to 5 minutes; audio is never padded or cropped')
    if detector is None:
        os.environ.setdefault('DETECT_INTERVAL', '120')
        from cartesia import _env, detect
        if not _env('CARTESIA_API_KEY'):
            raise ValueError('Set CARTESIA_API_KEY in the environment or .env before verification')
        detector = detect
    owned_workers = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    source_files = ['reconstruct_audio.py', 'reconstruction_worker.py', 'reconstruction_inference.py']
    if 'smoothed-vocoder' in presets:
        source_files.extend(('reconstruct_vocoder.py', 'vocoder_inference.py', 'vocoder_assets.json'))
    if 'filtered-prior' in presets:
        source_files.append('reconstruct_filtered_prior.py')
    if any(p in SPEECH_PRESETS for p in presets):
        source_files.extend(('reconstruct_speech_repair.py', 'speech_repair.py', 'speech_repair_assets.json', 'whisper_memory.py',
                             'reconstruct_filtered_prior.py', 'reconstruct_vocoder.py', 'vocoder_inference.py',
                             'vocoder_assets.json', 'reconstruct_vocoder_regions.py', 'vocoder_regions.py'))
    if any(p in CODEC_PRESETS for p in presets):
        source_files.extend(('reconstruct_codec.py', 'codec_inference.py', 'codec_content.py', 'codec_audio.py',
                             'codec_settings.py', 'codec_assets.json', 'speech_repair.py', 'speech_repair_assets.json',
                             'reconstruct_codec_band.py', 'reconstruct_vocoder.py', 'vocoder_inference.py', 'vocoder_assets.json'))
    source_files = list(dict.fromkeys(source_files))
    report = {'status': 'processing', 'source': str(source.resolve()), 'source_sha256': digest(source),
              'output': str(output.resolve()), 'presets': list(presets), 'seconds': seconds,
              'sample_rate': sample_rate, 'samples': len(audio), 'channels': channels,
              'reconstructor_code_sha256': digest(ROOT/'tools'/(
                  'reconstruct_vocoder.py' if presets[0] == 'smoothed-vocoder' else
                  'reconstruct_filtered_prior.py' if presets[0] == 'filtered-prior' else
                  'reconstruct_codec.py' if presets[0] in CODEC_PRESETS else
                  'reconstruct_speech_repair.py' if presets[0] in SPEECH_PRESETS else 'reconstruct_audio.py')),
              'verification_cli_code_sha256': digest(Path(__file__)),
              'source_files_sha256_at_start': {name: digest(ROOT/'tools'/name) for name in source_files},
              'boundary_safe_requested': boundary_safe,
              'required_negative_checks': confirmations,
              'quality_verified': False,
              'meaning': 'Exact output file checked negative by the current Cartesia detector; not proof of erasure or unchanged speech.',
              'attempts': []}
    save_report(report_path, report, create=True)
    published_identity = None
    try:
        with tempfile.TemporaryDirectory(prefix='.reconstruction-', dir=output.parent) as directory:
            directory = Path(directory)
            for index, preset in enumerate(presets):
                active_worker = worker
                if active_worker is None:
                    vocoder = preset == 'smoothed-vocoder'
                    filtered = preset == 'filtered-prior'
                    speech = preset in SPEECH_PRESETS
                    codec = preset in CODEC_PRESETS
                    key = 'codec' if codec else 'speech' if speech else 'vocoder' if vocoder else 'filtered' if filtered else 'flow'
                    if key not in owned_workers:
                        owned_workers[key] = ReconstructionWorker(
                            ROOT/('.venv/bin/python' if vocoder else '.cache/resemble-venv/bin/python'),
                            ROOT/'tools'/('reconstruct_vocoder.py' if vocoder else
                                         'reconstruct_filtered_prior.py' if filtered else
                                         'reconstruct_codec.py' if codec else
                                         'reconstruct_speech_repair.py' if speech else 'reconstruct_audio.py'),
                            boundary_safe=False if vocoder or filtered or speech or codec else boundary_safe)
                    active_worker = owned_workers[key]
                raw = directory/f'{index}.wav'
                metadata = active_worker.run(source, raw, preset, directory/f'{index}.json',
                                             timeout=max(300, seconds*8+120))
                if metadata.get('preset') != preset:
                    raise RuntimeError('Reconstruction metadata does not match the candidate')
                if preset in GUARDED_PRESETS and type(metadata.get('content_guard_passed')) is not bool:
                    raise RuntimeError('Guarded reconstruction did not return its required content assessment')
                if digest(source) != report['source_sha256']:
                    raise RuntimeError('Source changed while processing')
                if metadata.get('content_guard_passed') is False:
                    if raw.exists() or metadata.get('output_sha256') is not None or not metadata.get('rejection_reason'):
                        raise RuntimeError('Content-rejected reconstruction returned inconsistent publication metadata')
                    report['attempts'].append({'preset': preset, 'state': 'content_rejected', 'model': metadata})
                    save_report(report_path, report)
                    continue
                if metadata.get('output_sha256') != digest(raw):
                    raise RuntimeError('Reconstruction metadata does not match the candidate')
                edited, rate = sf.read(raw)
                if rate != sample_rate or edited.shape != audio.shape or not np.isfinite(edited).all():
                    raise RuntimeError('Reconstruction changed recording format or produced invalid samples')
                if np.max(np.abs(edited)) >= 1:
                    raise RuntimeError('Reconstruction is not below full scale')
                candidate = raw
                if output.suffix.lower() == '.flac':
                    candidate = directory/f'{index}.flac'
                    sf.write(candidate, edited, rate, format='FLAC', subtype='PCM_24')
                    decoded, decoded_rate = sf.read(candidate)
                    if (decoded_rate != rate or decoded.shape != edited.shape or
                            np.max(np.abs(decoded-edited)) > 2**-23):
                        raise RuntimeError('FLAC encoding changed recording format or exceeded its quantization limit')
                if candidate.stat().st_size > MAX_BYTES:
                    raise ValueError('The complete output exceeds the 32 MB detector limit; choose FLAC or a smaller recording')
                if preset in GUARDED_PRESETS:
                    evaluated, evaluated_rate = sf.read(candidate, dtype='float32')
                    scaled = evaluated*2**23
                    if (metadata.get('content_guard_sample_grid') != 'PCM_24' or
                            metadata.get('sample_rate') != evaluated_rate or
                            metadata.get('content_guard_pcm_sha256') != pcm_digest(evaluated) or
                            not np.array_equal(scaled, np.rint(scaled))):
                        raise RuntimeError('Published waveform differs from the PCM24 samples assessed by the content guard')
                if digest(source) != report['source_sha256']:
                    raise RuntimeError('Source changed while processing')
                sha = digest(candidate)
                attempt = {'preset': preset, 'sha256': sha, 'model': metadata, 'state': 'checking'}
                report['attempts'].append(attempt)
                save_report(report_path, report)
                attempt['checks'] = []
                for number in range(confirmations):
                    check = {'ordinal': number+1, 'state': 'checking'}
                    attempt['checks'].append(check)
                    save_report(report_path, report)
                    status, response = detector(candidate)
                    if digest(candidate) != sha:
                        raise RuntimeError('Candidate changed during verification')
                    check['http_status'] = attempt['http_status'] = status
                    if status != 200 or not isinstance(response, dict) or type(response.get('detected')) is not bool:
                        raise RuntimeError(f'Verification unavailable (HTTP {status}); no output published')
                    check.update(detected=response['detected'], state='checked',
                                 checked_at=datetime.now(timezone.utc).isoformat())
                    attempt.update(detected=check['detected'], state='checked', checked_at=check['checked_at'])
                    save_report(report_path, report)
                    if check['detected']:
                        break
                if attempt['detected']:
                    continue
                if digest(source) != report['source_sha256']:
                    raise RuntimeError('Source changed during verification')
                published_identity = publish_exact(candidate, output, sha)
                report.update(status='verified_negative', output_sha256=sha, selected_preset=preset,
                              checked_at=attempt['checked_at'], negative_checks=len(attempt['checks']))
                save_report(report_path, report)
                return report
        report['status'] = 'not_cleared'
        save_report(report_path, report)
        message = ('No candidate passed the requested content and detector checks; no output published'
                   if any(a['state'] == 'content_rejected' for a in report['attempts']) else
                   'All candidate outputs remained detected; no output published')
        raise NotCleared(message)
    except NotCleared:
        raise
    except Exception as error:
        if published_identity is not None:
            remove_own_output(output, published_identity)
        report.update(status='verification_failed', error_type=type(error).__name__)
        save_report(report_path, report)
        raise
    finally:
        for owned in owned_workers.values():
            owned.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--presets', nargs='+', choices=PRESETS, default=['full-flow', 'strong'])
    parser.add_argument('--report', type=Path)
    parser.add_argument('--boundary-safe', action='store_true', help='Use experimental corrected chunk scheduling')
    parser.add_argument('--confirmations', type=int, choices=(1, 2, 3), default=2,
                        help='Required negative responses for the same exact file (default: 2)')
    args = parser.parse_args()
    report_path = args.report or ROOT/'.cache/verification'/(args.output.stem+'.'+uuid.uuid4().hex+'.json')
    report_existed = report_path.exists()
    try:
        report = reconstruct_verified(args.input, args.output, report_path, tuple(args.presets),
                                      boundary_safe=args.boundary_safe, confirmations=args.confirmations)
    except NotCleared as error:
        print(str(error), file=sys.stderr)
        print('Audit report:', report_path, file=sys.stderr)
        return 2
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        print('No verified output was published.', file=sys.stderr)
        if not report_existed and report_path.exists():
            print('Audit report:', report_path, file=sys.stderr)
        return 3
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'sha256': report['output_sha256'], 'preset': report['selected_preset'],
                      'negative_checks': report['negative_checks'],
                      'quality_verified': False, 'report': str(report_path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
