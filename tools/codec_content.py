"""Prospective dual guards with local, language-specific secondary models."""
from importlib.metadata import version
import json
from math import gcd

from scipy.signal import resample_poly
import soundfile as sf

from codec_audio import validate_audio
from codec_inference import ROOT, assets, check_environment, verify_files
from codec_settings import LANGUAGE_MAX_DISAGREEMENT, MAX_DISAGREEMENT, MMS_CODES
from codec_text import paired_error, text_units
from whisper_encoder_reuse import prepare_original

SENSE_LANGUAGES = frozenset(('en', 'ja', 'ko', 'zh'))
LARGE_LANGUAGES = frozenset(('hi', 'ta'))


def secondary_for(language):
    return 'fastconformer' if language == 'ar' else 'sensevoice' if language in SENSE_LANGUAGES else 'mms'


def assess(before, after):
    secondary = before.get('secondary', 'mms')
    primary = before.get('primary', 'turbo')
    if secondary not in ('mms', 'sensevoice', 'fastconformer') or primary not in ('turbo', 'large'):
        raise ValueError('Unknown secondary content guard')
    names = (primary, secondary)
    scores = {name: paired_error(before.get(name, []), after.get(name, []), before.get('language')) for name in names}
    same_profile = (after.get('primary', primary) == primary and after.get('secondary', secondary) == secondary and
                    after.get('language') == before.get('language'))
    maximum = LANGUAGE_MAX_DISAGREEMENT.get(before.get('language'), MAX_DISAGREEMENT)
    passed = same_profile and all(score is not None and score <= maximum+1e-12 for score in scores.values())
    return {'content_guard_passed': passed, 'paired_disagreement': scores, 'primary_guard': primary,
            'secondary_guard': secondary, 'maximum_paired_disagreement': maximum,
            'original_units': {name: len(before.get(name, [])) for name in names},
            'processed_units': {name: len(after.get(name, [])) for name in names}}


class ContentRecognizer:
    def __init__(self):
        check_environment()
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        whisper = json.loads((ROOT/'tools/speech_repair_assets.json').read_text())
        directory = ROOT/whisper['local_directory']
        verify_files(directory, whisper['files_sha256'])
        self.entries = assets()
        self.mms_directory = ROOT/self.entries['mms']['local_directory']
        torch.set_num_threads(4)
        self.whisper_processor = WhisperProcessor.from_pretrained(directory, local_files_only=True)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            directory, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
        self.whisper_model.generation_config.forced_decoder_ids = None
        self.mms_processor = self.mms_model = self.mms_language = None
        self.large_processor = self.large_model = self.fast = None
        self.sense = self.sense_language = None
        self.normalizer = BasicTextNormalizer(remove_diacritics=False)
        self.metadata = {'guard_version': 'prospective-v3',
            'turbo': {'model': whisper['model_id'], 'revision': whisper['revision']},
            'primary_profiles': {language: 'large' if language in LARGE_LANGUAGES else 'turbo' for language in MMS_CODES},
            'secondary_profiles': {language: secondary_for(language) for language in MMS_CODES},
            'large': {k: self.entries['large'][k] for k in ('model_id', 'revision', 'license')},
            'fastconformer': {k: self.entries['fastconformer'][k] for k in ('model_id', 'revision', 'upstream', 'license', 'artifact')},
            'mms': {k: self.entries['mms'][k] for k in ('model_id', 'revision', 'license')},
            'sensevoice': {k: self.entries['sensevoice'][k] for k in ('model_id', 'export', 'license', 'license_url')},
            'attention': 'Whisper/MMS SDPA', 'precision': 'Whisper/MMS/SenseVoice float32; explicitly pinned FastConformer INT8 export', 'threads': 4,
            'language_policy': 'Detect original language, force it for candidates; eleven supported utterance languages',
            'supported_languages': sorted(MMS_CODES),
            'normalization': 'NFKC; Hindi/Tamil characters preserving combining vowel signs and viramas, punctuation/symbols removed, one-percent paired limit; Arabic vowel marks/tatweel removed; BasicTextNormalizer(False) elsewhere; Chinese OpenCC1.1.9 t2s characters; Japanese characters; Korean KsponSpeech spacing-normalized words; whitespace words otherwise',
            'short_source_encoder_reuse': True, 'transcripts_written_to_disk': False,
            'candidate_guard_order': 'Secondary first; a failed secondary skips the primary, and accepted outputs still require both recognizers',
            'model_downloads_during_processing': False}

    def check_profiles(self):
        verify_files(self.mms_directory, self.entries['mms']['files_sha256'])
        entry = self.entries['sensevoice']
        verify_files(ROOT/entry['local_directory'], entry['files_sha256'])
        for name in ('large', 'fastconformer'):
            entry = self.entries[name]
            verify_files(ROOT/entry['local_directory'], entry['files_sha256'])
        if version('sherpa-onnx') != entry['sherpa_onnx_version'] or version('OpenCC') != '1.1.9':
            raise RuntimeError('Install the pinned secondary recognizer and text normalization dependencies')

    @staticmethod
    def audio(path):
        audio, rate = sf.read(path, dtype='float32')
        validate_audio(audio, rate)
        audio = audio.mean(1) if audio.ndim == 2 else audio
        divisor = gcd(rate, 16000)
        return resample_poly(audio, 16000//divisor, rate//divisor)

    def features(self, path):
        audio = self.audio(path)
        batch = self.whisper_processor(audio, sampling_rate=16000, return_tensors='pt',
                                       return_attention_mask=True, truncation=False, padding='max_length',
                                       max_length=max(len(audio), 30*16000))
        return audio, batch

    def _mms(self, audio, language):
        import torch
        if self.mms_model is None:
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            verify_files(self.mms_directory, self.entries['mms']['files_sha256'])
            self.mms_processor = AutoProcessor.from_pretrained(self.mms_directory, local_files_only=True)
            self.mms_model = Wav2Vec2ForCTC.from_pretrained(
                self.mms_directory, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
        adapter = MMS_CODES[language]
        self.mms_processor.tokenizer.set_target_lang(adapter)
        if self.mms_language != adapter:
            self.mms_model.load_adapter(adapter, local_files_only=True, use_safetensors=True)
            self.mms_model.eval()
            self.mms_language = adapter
        batch = self.mms_processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        with torch.inference_mode():
            prediction = self.mms_model(**batch).logits.argmax(-1)
        decoded = self.mms_processor.tokenizer.decode(prediction[0], output_char_offsets=True, output_word_offsets=True)
        if decoded.text != self.mms_processor.batch_decode(prediction)[0]:
            raise RuntimeError('MMS offset decoding changed ordinary recognition')
        factor = self.mms_model.config.inputs_to_logits_ratio/self.mms_processor.feature_extractor.sampling_rate
        characters = language in ('ja', 'zh')
        offsets = decoded.char_offsets if characters else decoded.word_offsets
        alignment = {'text': decoded.text,
            'chunks': [{'text': value['char' if characters else 'word'],
                        'timestamp': (value['start_offset']*factor, value['end_offset']*factor)} for value in offsets]}
        return decoded.text, alignment

    def _sense(self, audio, language):
        entry = self.entries['sensevoice']
        if self.sense is None or self.sense_language != language:
            import sherpa_onnx
            if version('sherpa-onnx') != entry['sherpa_onnx_version']:
                raise RuntimeError('Install the pinned sherpa-onnx version')
            directory = ROOT/entry['local_directory']
            verify_files(directory, entry['files_sha256'])
            self.sense = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(directory/'model.onnx'), tokens=str(directory/'tokens.txt'),
                num_threads=4, language=language, use_itn=False, provider='cpu', debug=False)
            self.sense_language = language
        stream = self.sense.create_stream()
        stream.accept_waveform(16000, audio)
        self.sense.decode_stream(stream)
        return stream.result.text, None

    def _fast(self, audio):
        if self.fast is None:
            import sherpa_onnx
            entry = self.entries['fastconformer']
            if version('sherpa-onnx') != entry['sherpa_onnx_version']:
                raise RuntimeError('Install the pinned sherpa-onnx version')
            folder = ROOT/entry['local_directory']
            verify_files(folder, entry['files_sha256'])
            self.fast = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(folder/'model.int8.onnx'), tokens=str(folder/'tokens.txt'), num_threads=4,
                sample_rate=16000, feature_dim=80, decoding_method='greedy_search', provider='cpu')
        stream = self.fast.create_stream()
        stream.accept_waveform(16000, audio)
        self.fast.decode_stream(stream)
        return stream.result.text, None

    def _large(self):
        if self.large_model is None:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            entry = self.entries['large']
            folder = ROOT/entry['local_directory']
            verify_files(folder, entry['files_sha256'])
            self.large_processor = WhisperProcessor.from_pretrained(folder, local_files_only=True)
            self.large_model = WhisperForConditionalGeneration.from_pretrained(
                folder, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
            self.large_model.generation_config.forced_decoder_ids = None
        return self.large_processor, self.large_model

    def secondary_transcribe(self, path, language, audio=None):
        audio = self.audio(path) if audio is None else audio
        name = secondary_for(language)
        text, alignment = (self._sense(audio, language) if name == 'sensevoice' else
                           self._fast(audio) if name == 'fastconformer' else self._mms(audio, language))
        if alignment is not None:
            info = sf.info(path)
            alignment['duration'] = info.frames/info.samplerate
        return name, text_units(text, self.normalizer, language), alignment

    def transcribe_candidate(self, path, before):
        return self.transcribe(path, before['language'], comparison=before)

    def transcribe(self, path, language=None, comparison=None):
        import torch
        audio, batch = self.features(path)
        generation, reused = dict(batch), False
        with torch.inference_mode():
            if language is None:
                language, generation, reused = prepare_original(self.whisper_model, batch)
            if language not in MMS_CODES:
                return {'language': language, 'supported': False, 'turbo': [], 'mms': [], 'secondary': 'mms'}
            early_secondary = None
            if comparison is not None:
                primary = 'large' if language in LARGE_LANGUAGES else 'turbo'
                secondary = secondary_for(language)
                if (comparison.get('language') != language or comparison.get('primary', 'turbo') != primary or
                        comparison.get('secondary', 'mms') != secondary):
                    raise ValueError('Candidate comparison requires the same source language profile')
                early_secondary = self.secondary_transcribe(path, language, audio)
                if early_secondary[0] != secondary:
                    raise RuntimeError('Candidate secondary recognizer changed profile')
                score = paired_error(comparison.get(secondary, []), early_secondary[1], language)
                maximum = LANGUAGE_MAX_DISAGREEMENT.get(language, MAX_DISAGREEMENT)
                if score is None or score > maximum+1e-12:
                    return {'language': language, 'supported': True, 'primary': primary, 'secondary': secondary,
                            primary: [], secondary: early_secondary[1], 'alignment': early_secondary[2],
                            'source_encoder_reused': False, 'primary_skipped': True,
                            'primary_skip_reason': 'secondary_recognizer_rejected_candidate'}
            primary, processor, model = 'turbo', self.whisper_processor, self.whisper_model
            if language in LARGE_LANGUAGES:
                primary, reused = 'large', False
                processor, model = self._large()
                generation = processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True,
                                       truncation=False, padding='max_length', max_length=max(len(audio), 30*16000))
            prediction = model.generate(
                **generation, language=language, task='transcribe', return_timestamps=True,
                max_new_tokens=440, do_sample=False, condition_on_prev_tokens=False)
            text = processor.batch_decode(prediction, skip_special_tokens=True)[0]
            secondary, secondary_units, alignment = (early_secondary if early_secondary is not None else
                                                     self.secondary_transcribe(path, language, audio))
        return {'language': language, 'supported': True, 'primary': primary, 'secondary': secondary,
                primary: text_units(text, self.normalizer, language), secondary: secondary_units,
                'alignment': alignment, 'source_encoder_reused': reused}
