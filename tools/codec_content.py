"""Prospective dual guards with local, language-specific secondary models."""
from importlib.metadata import version
import json
from math import gcd

from scipy.signal import resample_poly
import soundfile as sf

from codec_audio import validate_audio
from codec_inference import ROOT, assets, check_environment, verify_files
from codec_settings import MAX_DISAGREEMENT, MMS_CODES
from codec_text import paired_error, text_units
from whisper_encoder_reuse import prepare_original

SENSE_LANGUAGES = frozenset(('en', 'ja', 'ko', 'zh'))


def assess(before, after):
    secondary = before.get('secondary', 'mms')
    if secondary not in ('mms', 'sensevoice'):
        raise ValueError('Unknown secondary content guard')
    names = ('turbo', secondary)
    scores = {name: paired_error(before.get(name, []), after.get(name, []), before.get('language')) for name in names}
    same_profile = after.get('secondary', secondary) == secondary and after.get('language') == before.get('language')
    passed = same_profile and all(score is not None and score <= MAX_DISAGREEMENT+1e-12 for score in scores.values())
    return {'content_guard_passed': passed, 'paired_disagreement': scores, 'secondary_guard': secondary,
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
        self.sense = self.sense_language = None
        self.normalizer = BasicTextNormalizer(remove_diacritics=False)
        self.metadata = {'guard_version': 'prospective-v2',
            'turbo': {'model': whisper['model_id'], 'revision': whisper['revision']},
            'secondary_profiles': {language: 'sensevoice' if language in SENSE_LANGUAGES else 'mms' for language in MMS_CODES},
            'mms': {k: self.entries['mms'][k] for k in ('model_id', 'revision', 'license')},
            'sensevoice': {k: self.entries['sensevoice'][k] for k in ('model_id', 'export', 'license', 'license_url')},
            'attention': 'Whisper/MMS SDPA', 'precision': 'float32', 'threads': 4,
            'language_policy': 'Detect original language, force it for candidates; eleven supported utterance languages',
            'supported_languages': sorted(MMS_CODES),
            'normalization': 'NFKC BasicTextNormalizer(False); Chinese OpenCC1.1.9 t2s characters; Japanese characters; Korean KsponSpeech spacing-normalized word comparisons; whitespace words otherwise',
            'short_source_encoder_reuse': True, 'transcripts_written_to_disk': False,
            'model_downloads_during_processing': False}

    def check_profiles(self):
        verify_files(self.mms_directory, self.entries['mms']['files_sha256'])
        entry = self.entries['sensevoice']
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

    def secondary_transcribe(self, path, language, audio=None):
        audio = self.audio(path) if audio is None else audio
        name = 'sensevoice' if language in SENSE_LANGUAGES else 'mms'
        text, alignment = self._sense(audio, language) if name == 'sensevoice' else self._mms(audio, language)
        if alignment is not None:
            info = sf.info(path)
            alignment['duration'] = info.frames/info.samplerate
        return name, text_units(text, self.normalizer, language), alignment

    def transcribe(self, path, language=None):
        import torch
        audio, batch = self.features(path)
        generation, reused = dict(batch), False
        with torch.inference_mode():
            if language is None:
                language, generation, reused = prepare_original(self.whisper_model, batch)
            if language not in MMS_CODES:
                return {'language': language, 'supported': False, 'turbo': [], 'mms': [], 'secondary': 'mms'}
            prediction = self.whisper_model.generate(
                **generation, language=language, task='transcribe', return_timestamps=True,
                max_new_tokens=440, do_sample=False, condition_on_prev_tokens=False)
            text = self.whisper_processor.batch_decode(prediction, skip_special_tokens=True)[0]
            secondary, secondary_units, alignment = self.secondary_transcribe(path, language, audio)
        return {'language': language, 'supported': True, 'secondary': secondary,
                'turbo': text_units(text, self.normalizer, language), secondary: secondary_units,
                'alignment': alignment, 'source_encoder_reused': reused}
