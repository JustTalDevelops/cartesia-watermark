"""Two pinned, local recognizers for the experimental codec content guard."""
import copy
import json
from math import gcd
from pathlib import Path

from scipy.signal import resample_poly
import soundfile as sf

from codec_inference import ROOT, assets, check_environment, verify_files
from codec_settings import MAX_DISAGREEMENT, MMS_CODES
from speech_repair import disagreement, units


def assess(before, after):
    scores = {name: disagreement(before[name], after[name]) for name in ('turbo', 'mms')}
    passed = all(score is not None and score <= MAX_DISAGREEMENT+1e-12 for score in scores.values())
    return {'content_guard_passed': passed, 'paired_disagreement': scores,
            'original_units': {name: len(before[name]) for name in scores},
            'processed_units': {name: len(after[name]) for name in scores}}


class ContentRecognizer:
    def __init__(self):
        check_environment()
        import torch
        from transformers import (AutoProcessor, Wav2Vec2ForCTC, WhisperProcessor,
                                  WhisperForConditionalGeneration)
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        whisper = json.loads((ROOT/'tools/speech_repair_assets.json').read_text())
        whisper_directory = ROOT/whisper['local_directory']
        verify_files(whisper_directory, whisper['files_sha256'])
        mms = assets()['mms']
        self.mms_directory = ROOT/mms['local_directory']
        verify_files(self.mms_directory, mms['files_sha256'])
        torch.set_num_threads(4)
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_directory, local_files_only=True)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            whisper_directory, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
        self.whisper_model.generation_config.forced_decoder_ids = None
        self.mms_processor = AutoProcessor.from_pretrained(self.mms_directory, local_files_only=True)
        self.mms_model = Wav2Vec2ForCTC.from_pretrained(
            self.mms_directory, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
        self.mms_language = None
        self.normalizer = BasicTextNormalizer(remove_diacritics=False)
        self.metadata = {'turbo': {'model': whisper['model_id'], 'revision': whisper['revision']},
                         'mms': {'model': mms['model_id'], 'revision': mms['revision'], 'license': mms['license']},
                         'attention': 'sdpa', 'precision': 'float32', 'threads': 4,
                         'language_policy': 'Detect from original initial Whisper features, force for all candidates; reject unsupported adapters',
                         'supported_languages': sorted(MMS_CODES),
                         'normalization': 'BasicTextNormalizer(remove_diacritics=False), NFKC; Chinese/Japanese characters, otherwise whitespace units',
                         'transcripts_written_to_disk': False, 'model_downloads_during_processing': False}

    def features(self, path):
        audio, rate = sf.read(path, dtype='float32')
        audio = audio.mean(1) if audio.ndim == 2 else audio
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000//divisor, rate//divisor)
        batch = self.whisper_processor(audio, sampling_rate=16000, return_tensors='pt',
                                       return_attention_mask=True, truncation=False, padding='max_length',
                                       max_length=max(len(audio), 30*16000))
        return audio, batch

    def transcribe(self, path, language=None):
        import torch
        audio, batch = self.features(path)
        with torch.inference_mode():
            if language is None:
                configuration = copy.deepcopy(self.whisper_model.generation_config)
                configuration.language = None
                token = int(self.whisper_model.detect_language(
                    input_features=batch.input_features, generation_config=configuration)[0])
                tag = next((name for name, value in configuration.lang_to_id.items() if value == token), '')
                language = tag[2:-2] if tag.startswith('<|') and tag.endswith('|>') else None
            if language not in MMS_CODES:
                return {'language': language, 'supported': False, 'turbo': [], 'mms': []}
            prediction = self.whisper_model.generate(
                **batch, language=language, task='transcribe', return_timestamps=True,
                max_new_tokens=440, do_sample=False, condition_on_prev_tokens=False)
            turbo_text = self.whisper_processor.batch_decode(prediction, skip_special_tokens=True)[0]
            adapter = MMS_CODES[language]
            self.mms_processor.tokenizer.set_target_lang(adapter)
            if self.mms_language != adapter:
                self.mms_model.load_adapter(adapter, local_files_only=True, use_safetensors=True)
                self.mms_model.eval()
                self.mms_language = adapter
            inputs = self.mms_processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            logits = self.mms_model(**inputs).logits
            mms_text = self.mms_processor.batch_decode(logits.argmax(-1))[0]
        return {'language': language, 'supported': True,
                'turbo': units(turbo_text, self.normalizer, language),
                'mms': units(mms_text, self.normalizer, language)}
