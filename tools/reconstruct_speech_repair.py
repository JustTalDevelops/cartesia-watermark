"""Experimental local reconstruction with a recognizer-consistency guard.

Rebuild only mismatched speech regions with a regional vocoder donor. Final
samples are quantized before recognition, so WAV/FLAC publication preserves the
waveform assessed by the guard. Recognizer agreement is not proof of unchanged
speech. Cartesia checks are performed separately by verify_reconstruction.py.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from math import gcd
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from reconstruction_worker import ReconstructionWorker
from speech_repair import (PRESETS, MAX_COVERAGE, MAX_DISAGREEMENT, AlignmentUnavailable,
                           RepairTooLarge, aligned_regions, blend_regions, disagreement,
                           language_from_chunks, pcm_digest, units)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = json.loads((ROOT/'tools/speech_repair_assets.json').read_text())
SETTINGS = {'maximum_asr_disagreement': MAX_DISAGREEMENT, 'maximum_patch_coverage': MAX_COVERAGE,
            'region_padding_seconds': .08, 'crossfade_seconds': .04, 'donor_mix': 1.,
            'donor': 'Unscaled sigma1 BigVGAN, selected global blocks, then shared output peak gain',
            'sample_grid_before_guard': 'PCM_24', 'threads': 4}
PINNED_FILES = {
    'reconstruct_audio.py': '25f3e120598995e398fe76a68a3fda943b53e4f294cad5887207888fb9229c96',
    'reconstruction_inference.py': '1a3c6b8ae05019f04b9650a11b81a331b4ef6b11e9dc43c34e95a0371cbf8a8f',
    'reconstruct_filtered_prior.py': '39e9709e619090b87a52bf017f86e454b7653d1931059361f7b465cbce25195b',
    'reconstruct_vocoder.py': '3378845287c3f1bf874147392184c5b15b43ca954adacd6eac484715e4564db0',
    'vocoder_inference.py': '38d02df4aa5d488aab7b37320dcbf6cc6e980178e23721783145adef138c37f2',
    'vocoder_assets.json': 'ea7870e32da6336a800a3ddbe555d527267124470b44f8e7e0c72603aae4e7f6',
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class WordRecognizer:
    def __init__(self):
        from importlib.metadata import version
        if sys.version_info[:2] != (3, 12) or version('transformers') != '4.46.3':
            raise RuntimeError('Use the pinned Python 3.12 speech-repair environment')
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
        directory = ROOT/ASSETS['local_directory']
        for name, expected in ASSETS['files_sha256'].items():
            if not (directory/name).is_file() or digest(directory/name) != expected:
                raise RuntimeError('Install the pinned local speech-repair ASR assets before processing')
        torch.set_num_threads(4)
        processor = WhisperProcessor.from_pretrained(directory, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            directory, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
        model.generation_config.forced_decoder_ids = None
        from whisper_memory import install_encoder_attention_reduction
        self.attention_handles = install_encoder_attention_reduction(model)
        self.pipeline = pipeline('automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
                                 feature_extractor=processor.feature_extractor, device=-1, torch_dtype=torch.float32)
        self.normalizer = BasicTextNormalizer(remove_diacritics=False)
        self.language_codes = TO_LANGUAGE_CODE
        self.metadata = {'model': ASSETS['model_id'], 'revision': ASSETS['revision'],
                         'checkpoint_sha256': ASSETS['files_sha256']['model.safetensors'],
                         'precision': 'float32', 'attention': 'eager', 'transformers': version('transformers'),
                         'encoder_attention_diagnostics_retained': False,
                         'decoder_cross_attention_for_word_alignment': 'unchanged',
                         'language_policy': 'Use unambiguous chunk tags; when absent, detect from initial features. Force original language for candidates.',
                         'normalization': 'BasicTextNormalizer(remove_diacritics=False), NFKC; Chinese/Japanese/Cantonese characters, otherwise whitespace units'}

    def transcribe(self, path, language=None):
        audio, rate = sf.read(path, dtype='float32')
        audio = audio.mean(1) if audio.ndim == 2 else audio
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000//divisor, rate//divisor)
        result = self.pipeline({'raw': audio, 'sampling_rate': 16000}, return_timestamps='word',
                               return_language=True, generate_kwargs={
                                   'language': language, 'task': 'transcribe', 'do_sample': False,
                                   'condition_on_prev_tokens': False, 'max_new_tokens': 440})
        if language is None:
            language, needs_detection = language_from_chunks(result, self.language_codes)
            if needs_detection and result.get('text', '').strip():
                # Long-form generation strips language prompt tokens before
                # word decoding. Reuse Whisper's own initial-feature detector
                # rather than treating missing output tags as missing speech.
                import torch
                features = self.pipeline.feature_extractor(
                    audio, sampling_rate=16000, return_tensors='pt', truncation=False,
                    padding='max_length', max_length=max(len(audio), 30*16000))
                with torch.inference_mode():
                    model = self.pipeline.model
                    token = int(model.detect_language(input_features=features.input_features,
                                                     generation_config=model.generation_config)[0])
                tag = next((name for name, value in model.generation_config.lang_to_id.items()
                            if value == token), '')
                code = tag[2:-2] if tag.startswith('<|') and tag.endswith('|>') else None
                language = code if code in self.language_codes.values() else None
                result['language_detected_from_initial_features'] = language is not None
        return result, language


def validate_paths(source, output, metadata):
    if source.resolve() == output.resolve() or output.exists() or output.suffix.lower() != '.wav':
        raise ValueError('Choose a new WAV output separate from the input')
    if metadata and (metadata.exists() or metadata.resolve() in (source.resolve(), output.resolve())):
        raise ValueError('Choose a new metadata path separate from the audio files')


def quantize(path, audio, rate):
    sf.write(path, audio, rate, format='FLAC', subtype='PCM_24')
    decoded, actual_rate = sf.read(path, dtype='float32')
    if actual_rate != rate or decoded.shape != audio.shape or np.max(np.abs(decoded-audio)) > 2**-23:
        raise RuntimeError('Quality encoding changed the recording format or exceeded PCM24 quantization')
    return decoded


class SpeechRepairRuntime:
    def __init__(self, recognizer=None, worker_factory=ReconstructionWorker):
        for name, expected in PINNED_FILES.items():
            if digest(ROOT/'tools'/name) != expected:
                raise RuntimeError('The pinned base reconstruction implementation changed')
        self.recognizer = recognizer if recognizer is not None else WordRecognizer()
        self.worker_factory = worker_factory
        self.workers = {}

    def worker(self, key):
        if key not in self.workers:
            script = {'filtered': 'reconstruct_filtered_prior.py', 'flow': 'reconstruct_audio.py',
                      'donor': 'reconstruct_vocoder_regions.py'}[key]
            python = ROOT/'.venv/bin/python' if key == 'donor' else Path(sys.executable)
            self.workers[key] = self.worker_factory(python, ROOT/'tools'/script,
                                                    boundary_safe=key == 'flow', isolated=False)
        return self.workers[key]

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()

    def process_file(self, source, output, preset='speech-repair', metadata=None):
        validate_paths(source, output, metadata)
        if preset not in PRESETS:
            raise ValueError('Unknown speech-repair preset')
        started = time.monotonic()
        original_sha = digest(source)
        original, rate = sf.read(source, dtype='float32')
        if (original.ndim not in (1, 2) or not np.isfinite(original).all() or
                (original.ndim == 2 and original.shape[1] not in (1, 2)) or
                not 8000 <= rate <= 96000 or not 5 <= len(original)/rate <= 300):
            raise ValueError('Choose complete finite mono/stereo speech, 5–300 seconds at 8–96 kHz')
        report = {'preset': preset, 'settings': SETTINGS, 'source_sha256': original_sha,
                  'sample_rate': rate, 'samples': len(original),
                  'channels': 1 if original.ndim == 1 else original.shape[1],
                  'asr': self.recognizer.metadata, 'content_guard_passed': False,
                  'quality_verified': False, 'output_sha256': None, 'repair_applied': False,
                  'meaning': 'Recognizer-consistency estimate only; no unchanged-speech, human approval or watermark-erasure guarantee'}
        before, language = self.recognizer.transcribe(source)
        report['language'] = language
        before_units = units(before['text'], self.recognizer.normalizer, language)
        report['original_units'] = len(before_units)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.speech-repair-', dir=output.parent) as temporary:
            directory = Path(temporary)
            if language is None or not before_units:
                report['rejection_reason'] = 'original_speech_not_recognized'
            else:
                base_preset = PRESETS[preset]
                key = 'filtered' if base_preset == 'filtered-prior' else 'flow'
                native = directory/'native.wav'
                base_info = self.worker(key).run(source, native, base_preset, directory/'native.json',
                                                 timeout=max(300, len(original)/rate*8+120))
                base, base_rate = sf.read(native, dtype='float32')
                if (base_info.get('output_sha256') != digest(native) or base_rate != rate or
                        base.shape != original.shape or not np.isfinite(base).all() or np.max(np.abs(base)) >= 1):
                    raise RuntimeError('Base reconstruction changed the format or returned invalid samples')
                report['base_model'] = base_info
                candidate = directory/'native.flac'
                result = quantize(candidate, base, rate)
                candidate_sha = digest(candidate)
                after, _ = self.recognizer.transcribe(candidate, language)
                if digest(candidate) != candidate_sha:
                    raise RuntimeError('Quality candidate changed during recognition')
                score = disagreement(before_units, units(after['text'], self.recognizer.normalizer, language))
                report['asr_initial_disagreement'] = report['asr_final_disagreement'] = score
                if score is not None and score > MAX_DISAGREEMENT+1e-12:
                    try:
                        regions = aligned_regions(before, after, len(original)/rate, self.recognizer.normalizer, language)
                        # Validate exact mask coverage before loading the donor model.
                        unused, coverage, _ = blend_regions(result, result, rate, regions)
                        del unused
                        manifest = directory/'donor.input.json'
                        manifest.write_text(json.dumps({'source': str(source.resolve()), 'source_sha256': original_sha,
                                                        'regions_seconds': regions}))
                        donor_path = directory/'donor.wav'
                        donor_info = self.worker('donor').run(manifest, donor_path, 'regional-vocoder',
                                                              directory/'donor.json', timeout=max(300, len(original)/rate*8+120))
                        donor, donor_rate = sf.read(donor_path, dtype='float32')
                        if (donor_info.get('output_sha256') != digest(donor_path) or donor_rate != rate or
                                donor_info.get('source_sha256') != original_sha or
                                donor_info.get('valid_regions_seconds') != [list(r) for r in regions] or
                                donor_info.get('waveform_is_partial') is not True or donor.shape != original.shape):
                            raise RuntimeError('Regional donor does not match the requested source and speech intervals')
                        patched, coverage, gain = blend_regions(result, donor, rate, regions)
                        candidate = directory/'repaired.flac'
                        result = quantize(candidate, patched, rate)
                        candidate_sha = digest(candidate)
                        repaired, _ = self.recognizer.transcribe(candidate, language)
                        if digest(candidate) != candidate_sha:
                            raise RuntimeError('Repaired candidate changed during recognition')
                        score = disagreement(before_units, units(repaired['text'], self.recognizer.normalizer, language))
                        report.update(repair_applied=True, donor_model=donor_info, regions_seconds=[list(r) for r in regions],
                                      patch_coverage=coverage, final_shared_gain=gain, asr_final_disagreement=score)
                    except (AlignmentUnavailable, RepairTooLarge) as error:
                        report['rejection_reason'] = type(error).__name__
                if score is not None and score <= MAX_DISAGREEMENT+1e-12:
                    report['content_guard_passed'] = True
                    report['content_guard_pcm_sha256'] = pcm_digest(result)
                    report['content_guard_sample_grid'] = 'PCM_24'
                else:
                    report.setdefault('rejection_reason', 'recognizer_disagreement')
                if digest(source) != original_sha:
                    raise RuntimeError('Source changed during speech repair')
                if report['content_guard_passed']:
                    with output.open('xb') as handle:
                        sf.write(handle, result, rate, format='WAV', subtype='FLOAT')
                    report['output_sha256'] = digest(output)
        if digest(source) != original_sha:
            raise RuntimeError('Source changed during speech repair')
        report['elapsed_seconds'] = time.monotonic()-started
        if metadata:
            metadata.parent.mkdir(parents=True, exist_ok=True)
            with metadata.open('x') as handle:
                json.dump(report, handle, indent=2)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--preset', choices=tuple(PRESETS), default='speech-repair')
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
        try:
            validate_paths(args.input, args.output, args.metadata)
        except ValueError as error:
            parser.error(str(error))
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        runtime = SpeechRepairRuntime()
        try:
            if args.worker:
                print(json.dumps({'ready': True}), file=protocol, flush=True)
                for line in sys.stdin:
                    try:
                        request = json.loads(line)
                        report = runtime.process_file(Path(request['input']), Path(request['output']),
                                                      request['preset'], Path(request['metadata']) if request.get('metadata') else None)
                        reply = {'report': report}
                    except Exception as error:
                        reply = {'error': f'{type(error).__name__}: {error}'}
                    print(json.dumps(reply), file=protocol, flush=True)
                return 0
            report = {'ready': True, 'presets': list(PRESETS), 'asr': runtime.recognizer.metadata} if args.check else runtime.process_file(
                args.input, args.output, args.preset, args.metadata)
        finally:
            runtime.close()
    print(json.dumps(report), flush=True)
    return 0 if report.get('content_guard_passed', True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
