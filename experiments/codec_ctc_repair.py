"""Develop localized codec repair without changing the frozen candidate.

The donor is a complete native reconstruction, never the original recording.
It must reproduce the original's entire normalized MMS sequence before any
repair is proposed. New output still needs both guards and whole-file detector
checks. Agreement here is neither human approval nor an erasure certificate.
"""
from math import gcd
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest, validate_audio
from codec_settings import MMS_CODES
from speech_repair import AlignmentUnavailable, RepairTooLarge, aligned_regions, blend_regions, units
from experiments.score_codec_study import TextScorer


class MMSOffsets(TextScorer):
    """Pinned FP32 MMS offsets, without target text.

    Offset conversion follows the upstream v4.46.3 Wav2Vec2 documentation:
    https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/wav2vec2
    """
    def __init__(self, cache):
        super().__init__('mms', cache)

    def aligned(self, path, language):
        import torch
        checksum = digest(path)
        audio, rate = sf.read(path, dtype='float32')
        validate_audio(audio, rate)
        duration = len(audio)/rate
        audio = audio.mean(1) if audio.ndim == 2 else audio
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000//divisor, rate//divisor)
        adapter = MMS_CODES[language]
        self.processor.tokenizer.set_target_lang(adapter)
        if self.language != adapter:
            self.model.load_adapter(adapter, local_files_only=True, use_safetensors=True)
            self.model.eval()
            self.language = adapter
        inputs = self.processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        with torch.inference_mode():
            prediction = self.model(**inputs).logits.argmax(-1)
        decoded = self.processor.tokenizer.decode(prediction[0], output_char_offsets=True, output_word_offsets=True)
        if decoded.text != self.processor.batch_decode(prediction)[0]:
            raise AlignmentUnavailable('Offset decoding changed the ordinary MMS transcription')
        seconds_per_frame = self.model.config.inputs_to_logits_ratio/self.processor.feature_extractor.sampling_rate
        characters = language in ('zh', 'ja')
        offsets = decoded.char_offsets if characters else decoded.word_offsets
        result = {'text': decoded.text, 'chunks': [
            {'text': value['char' if characters else 'word'],
             'timestamp': (value['start_offset']*seconds_per_frame, value['end_offset']*seconds_per_frame)}
            for value in offsets], 'source_sha256': checksum, 'duration': duration}
        annotated = [unit for chunk in result['chunks'] for unit in units(chunk['text'], self.normalize, language)]
        if annotated != units(result['text'], self.normalize, language) or not annotated:
            raise AlignmentUnavailable('CTC offsets do not cover the complete normalized sequence')
        if digest(path) != checksum:
            raise ValueError('Audio changed during CTC alignment')
        return result


def repair_from_native(original, native, band, native_audio, band_audio, rate, normalize, language,
                       padding=.12, maximum_coverage=.20, maximum_regions=8):
    """Make one bounded proposal; callers must reassess its exact PCM24 audio."""
    validate_audio(native_audio, rate)
    validate_audio(band_audio, rate)
    if native_audio.shape != band_audio.shape:
        raise ValueError('Native and band reconstructions must have the same complete sample grid')
    reference_units = units(original['text'], normalize, language)
    native_units = units(native['text'], normalize, language)
    if not reference_units or native_units != reference_units:
        raise AlignmentUnavailable('Native donor does not match the complete original MMS sequence')
    if units(band['text'], normalize, language) == reference_units:
        raise AlignmentUnavailable('No MMS mismatch needs a localized repair')
    duration = len(native_audio)/rate
    for result in (original, native, band):
        if abs(result['duration']-duration) > 1/rate:
            raise ValueError('Alignment and complete waveform durations differ')
    regions = aligned_regions(native, band, duration, normalize, language, padding=padding)
    if len(regions) > maximum_regions:
        raise RepairTooLarge('Too many distinct intervals need repair')
    output, coverage, gain = blend_regions(band_audio, native_audio, rate, regions,
                                          maximum_coverage=maximum_coverage)
    return np.asarray(output, dtype=np.float32), {'regions_seconds': regions, 'repaired_fraction': coverage,
        'gain': gain, 'padding_seconds': padding, 'maximum_coverage': maximum_coverage,
        'maximum_regions': maximum_regions, 'native_original_unit_parity': True,
        'quality_verified': False, 'content_guard_passed': False, 'api_checked': False,
        'meaning': 'One experimental proposal; requires assessment of the final PCM24 file with both guards.'}
