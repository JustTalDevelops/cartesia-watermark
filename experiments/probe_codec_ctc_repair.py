"""Reproduce a spent rejected band file and assess one local repair proposal.

No Cartesia requests, reference text, changed thresholds or publication as a
verified file. Run separately after the frozen cohort and its quality scoring.
Exact rejected/native FLAC hashes are checked before any new repair is made.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest, quantize, validate_audio
from codec_content import assess
from codec_inference import NativeCodec
from reconstruction_worker import ReconstructionWorker
from speech_repair import AlignmentUnavailable, RepairTooLarge, units
from experiments.codec_ctc_repair import MMSOffsets, repair_from_native
from experiments.score_codec_study import TextScorer


def probe(source, audit_path, directory):
    audit = json.loads(audit_path.read_text())
    source_sha = digest(source)
    if source_sha != audit['source_sha256'] or audit['status'] != 'not_cleared':
        raise ValueError('Use the exact spent source and its completed unsuccessful audit')
    native_attempt = next(a for a in audit['attempts'] if a['preset'] == 'codec-native')
    band_attempt = next(a for a in audit['attempts'] if a['preset'] == 'codec-band')
    if (native_attempt['model']['content_guard_passed'] is not True or
            band_attempt['state'] != 'content_rejected'):
        raise ValueError('This prototype needs a content-qualified native and content-rejected band file')
    language = native_attempt['model']['language']
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory/'result.json'
    report = {'source_sha256': source_sha, 'unsuccessful_audit_sha256': digest(audit_path),
              'probe_source_sha256': digest(Path(__file__)), 'repair_source_sha256': digest(ROOT/'experiments/codec_ctc_repair.py'),
              'language': language, 'reference_text_used': False, 'cartesia_requests': 0,
              'fresh_independent_sources': 0, 'frozen_study_result_changed': False,
              'quality_verified': False, 'complete': False}

    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(report_path)

    save()
    audio, rate = sf.read(source, dtype='float32')
    validate_audio(audio, rate)
    native_path, band_path = directory/'native.flac', directory/'band.flac'
    codec = NativeCodec()
    native_audio, gain = codec.reconstruct(audio, rate)
    quantize(native_path, native_audio, rate)
    expected_native = native_attempt['model']['native_codec_sha256']
    if digest(native_path) != expected_native or expected_native != band_attempt['model']['native_codec_sha256']:
        raise ValueError('Reproduced native differs from the spent measured file')
    manifest = directory/'band.input.json'
    manifest.write_text(json.dumps({'source': str(source.resolve()), 'source_sha256': source_sha,
                                    'native': str(native_path.resolve()), 'native_sha256': expected_native}))
    worker = ReconstructionWorker(ROOT/'.venv/bin/python', ROOT/'tools/reconstruct_codec_band.py')
    try:
        waveform = directory/'band.wav'
        info = worker.run(manifest, waveform, 'codec-band', directory/'band.json',
                          timeout=max(300, len(audio)/rate*8+120))
        band_audio, actual_rate = sf.read(waveform, dtype='float32')
        if (actual_rate != rate or band_audio.shape != audio.shape or
                info['output_sha256'] != digest(waveform)):
            raise ValueError('Band worker returned a changed complete grid')
        quantize(band_path, band_audio, rate)
    finally:
        worker.close()
    if digest(band_path) != band_attempt['model']['candidate_flac_sha256']:
        raise ValueError('Reproduced band differs from the exact rejected recording')
    report.update(native_sha256=expected_native, band_sha256=digest(band_path),
                  rejected_file_reproduction_exact=True, native_gain=gain)
    save()
    mms = MMSOffsets(directory/'unused-mms-cache')
    original = mms.aligned(source, language)
    native = mms.aligned(native_path, language)
    band = mms.aligned(band_path, language)
    native_audio, _ = sf.read(native_path, dtype='float32')
    band_audio, _ = sf.read(band_path, dtype='float32')
    try:
        repaired, proposal = repair_from_native(original, native, band, native_audio, band_audio,
                                                rate, mms.normalize, language)
    except (AlignmentUnavailable, RepairTooLarge) as error:
        report.update(complete=True, status='repair_unavailable', reason=str(error))
        save()
        return report
    repaired_path = directory/'repair.flac'
    quantize(repaired_path, repaired, rate)
    mms_after = mms.aligned(repaired_path, language)
    turbo = TextScorer('turbo', directory/'private-transcripts')
    turbo.cache.mkdir()
    turbo_before = turbo.transcribe(source, language)
    turbo_after = turbo.transcribe(repaired_path, language)
    assessment = assess({'turbo': units(turbo_before, turbo.normalize, language),
                         'mms': units(original['text'], mms.normalize, language)},
                        {'turbo': units(turbo_after, turbo.normalize, language),
                         'mms': units(mms_after['text'], mms.normalize, language)})
    if digest(source) != source_sha:
        raise ValueError('Original source changed during the development probe')
    report.update(complete=True, status='local_assessment_only', output_sha256=digest(repaired_path),
                  proposal=proposal, **assessment)
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        result = probe(args.source, args.audit, args.directory)
    print(json.dumps(result, indent=2), file=protocol)
