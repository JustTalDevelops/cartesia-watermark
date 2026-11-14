"""Score a sealed study's originals or final files without changing selection.

ASR metrics run in the pinned Python 3.12 environment; voice metrics use the
Python 3.13 evaluation environment. Raw transcripts/embeddings stay local.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import gcd
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from codec_inference import check_environment, verify_files
from codec_settings import MMS_CODES
from experiments.codec_evaluation_text import evaluation_units
from speech_repair import disagreement, units

WHISPER = {'small': ('openai/whisper-small', '973afd24965f72e36ca33b3055d56a652f456b4d'),
           'turbo': ('openai/whisper-large-v3-turbo', '41f01f3fe87f28c78e2fbf8b568835947dd65ed9')}


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class TextScorer:
    def __init__(self, metric, cache):
        check_environment()
        import torch
        from transformers import (AutoProcessor, Wav2Vec2ForCTC, WhisperProcessor,
                                  WhisperForConditionalGeneration)
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        torch.set_num_threads(4)
        self.metric, self.cache = metric, cache
        self.normalize = BasicTextNormalizer(remove_diacritics=False)
        self.language = None
        if metric == 'mms':
            assets = load(ROOT/'tools/codec_assets.json')['mms']
            self.folder = ROOT/assets['local_directory']
            verify_files(self.folder, assets['files_sha256'])
            self.processor = AutoProcessor.from_pretrained(self.folder, local_files_only=True)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.folder, local_files_only=True,
                                                      torch_dtype=torch.float32, attn_implementation='sdpa').eval()
            self.metadata = {'model': assets['model_id'], 'revision': assets['revision'], 'attention': 'sdpa',
                             'precision': 'float32', 'role': 'Also participates in the processing guard'}
        else:
            model, revision = WHISPER[metric]
            if metric == 'turbo':
                assets = load(ROOT/'tools/speech_repair_assets.json')
                self.folder = ROOT/assets['local_directory']
                verify_files(self.folder, assets['files_sha256'])
                parameters = {'local_files_only': True}
            else:
                assets = load(ROOT/'experiments/codec_evaluation_assets.json')['small']
                self.folder = ROOT/assets['local_directory']
                verify_files(self.folder, assets['files_sha256'])
                parameters = {'local_files_only': True}
            self.processor = WhisperProcessor.from_pretrained(self.folder, **parameters)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.folder, **parameters, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
            self.model.generation_config.forced_decoder_ids = None
            self.metadata = {'model': model, 'revision': revision, 'attention': 'sdpa', 'precision': 'float32',
                             'decode': 'Native timestamps, minimum30s features, complete longer input, max_new_tokens440, no sampling/previous-text conditioning',
                             'role': 'Evaluation-only checkpoint' if metric == 'small' else 'Also participates in the processing guard'}
        self.metadata['reference_normalization'] = 'NFKC, BasicTextNormalizer(False); Chinese OpenCC1.1.9 t2s.json before reference CER; Japanese CER; whitespace WER otherwise'
        self.metadata['paired_normalization'] = 'Original runtime normalization: NFKC BasicTextNormalizer(False), zh/ja characters, whitespace otherwise'
        self.configuration = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def transcribe(self, path, language):
        import torch
        checksum = digest(path)
        key = hashlib.sha256(json.dumps([checksum, language, self.configuration]).encode()).hexdigest()
        cache = self.cache/(key+'.json')
        if cache.exists():
            value = load(cache)
            if value['sha256'] != checksum or value['configuration_sha256'] != self.configuration:
                raise ValueError('Transcription cache metadata changed')
            return value['transcript']
        audio, rate = sf.read(path, dtype='float32')
        audio = audio.mean(1) if audio.ndim == 2 else audio
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000//divisor, rate//divisor)
        with torch.inference_mode():
            if self.metric == 'mms':
                adapter = MMS_CODES[language]
                self.processor.tokenizer.set_target_lang(adapter)
                if self.language != adapter:
                    self.model.load_adapter(adapter, local_files_only=True, use_safetensors=True)
                    self.model.eval()
                    self.language = adapter
                batch = self.processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
                prediction = self.model(**batch).logits.argmax(-1)
            else:
                batch = self.processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True,
                                       truncation=False, padding='max_length', max_length=max(len(audio), 30*16000))
                prediction = self.model.generate(**batch, language=language, task='transcribe', return_timestamps=True,
                                                 max_new_tokens=440, do_sample=False, condition_on_prev_tokens=False)
            text = (self.processor.batch_decode(prediction)[0] if self.metric == 'mms' else
                    self.processor.batch_decode(prediction, skip_special_tokens=True)[0])
        if digest(path) != checksum:
            raise ValueError('Audio changed during transcription')
        save(cache, {'sha256': checksum, 'configuration_sha256': self.configuration, 'language': language, 'transcript': text})
        return text

    def score(self, source, output, text, language):
        before_text = self.transcribe(source, language)
        before_raw = units(before_text, self.normalize, language)
        truth_raw = units(text, self.normalize, language)
        before = evaluation_units(before_text, language, self.normalize)
        truth = evaluation_units(text, language, self.normalize)
        error = disagreement(truth, before)
        row = {'original_reference_error': error, 'original_reference_error_raw_script': disagreement(truth_raw, before_raw),
               'original_reference_units': len(truth), 'original_recognized_units': len(before_raw),
               'original_calibration_eligible': error is not None and error <= .15+1e-12}
        if output is not None:
            after_text = self.transcribe(output, language)
            after_raw = units(after_text, self.normalize, language)
            after = evaluation_units(after_text, language, self.normalize)
            after_error = disagreement(truth, after)
            paired = disagreement(before_raw, after_raw)
            row.update(processed_reference_error=after_error, reference_error_delta=None if error is None or after_error is None else after_error-error,
                       processed_reference_error_raw_script=disagreement(truth_raw, after_raw),
                       asr_disagreement=paired, paired_disagreement=paired, processed_recognized_units=len(after_raw))
        return row


class VoiceScorer:
    def __init__(self, cache):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        from eval_voice_quality import VoiceQuality
        torch.set_num_threads(2)
        model, revision = 'microsoft/wavlm-base-plus-sv', 'feb593a6c23c1cc3d9510425c29b0a14d2b07b1e'
        assets = load(ROOT/'experiments/codec_evaluation_assets.json')['speaker']
        folder = ROOT/assets['local_directory']
        verify_files(folder, assets['files_sha256'])
        self.feature = Wav2Vec2FeatureExtractor.from_pretrained(folder, local_files_only=True)
        self.model = WavLMForXVector.from_pretrained(folder, local_files_only=True).eval()
        self.mos = VoiceQuality()
        self.cache = cache
        self.metadata = {'speaker_model': model, 'speaker_revision': revision, 'dnsmos': self.mos.metadata,
                         'window_seconds': 10, 'window_centers_fraction': [.2, .5, .8], 'threads': 2,
                         'meaning': 'Uncalibrated embedding/MOS estimates, not human listening approval'}
        self.configuration = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def features(self, path):
        import torch
        from remover import resample
        checksum = digest(path)
        key = hashlib.sha256(json.dumps([checksum, self.configuration]).encode()).hexdigest()
        cache = self.cache/(key+'.json')
        if cache.exists():
            return load(cache)
        audio, rate = sf.read(path)
        audio = audio.mean(1) if audio.ndim == 2 else audio
        audio = resample(audio, rate, 16000).astype(np.float32)
        length = 10*16000
        starts = sorted(set(max(0, min(len(audio)-length, int(f*len(audio)-length/2))) for f in (.2, .5, .8)))
        embeddings = []
        with torch.inference_mode():
            for start in starts:
                batch = self.feature(audio[start:start+length], sampling_rate=16000, return_tensors='pt')
                vector = self.model(**batch).embeddings[0].double().numpy()
                vector /= np.linalg.norm(vector)
                if not np.isfinite(vector).all():
                    raise ValueError('Invalid speaker embedding')
                embeddings.append(vector.tolist())
        result = {'window_starts_seconds': [s/16000 for s in starts], 'embeddings': embeddings, 'dnsmos': self.mos.score(path)}
        if digest(path) != checksum:
            raise ValueError('Audio changed during voice scoring')
        save(cache, result)
        return result

    def score(self, source, output, text, language):
        before = self.features(source)
        row = {'original_dnsmos': before['dnsmos'], 'window_starts_seconds': before['window_starts_seconds']}
        if output is not None:
            after = self.features(output)
            if before['window_starts_seconds'] != after['window_starts_seconds']:
                raise ValueError('Output has different voice-scoring windows')
            cosines = [float(np.clip(np.asarray(a)@np.asarray(b), -1, 1)) for a, b in zip(before['embeddings'], after['embeddings'])]
            row.update(speaker_window_cosines=cosines, speaker_minimum=min(cosines), processed_dnsmos=after['dnsmos'],
                       dnsmos_delta={k: after['dnsmos'][k]-before['dnsmos'][k] for k in ('signal', 'background', 'overall')})
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='Materialized frozen requests plus generation receipts; output paths may be absent')
    parser.add_argument('--metric', choices=('small', 'turbo', 'mms', 'voice'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--originals-only', action='store_true')
    args = parser.parse_args()
    plan_sha = digest(args.plan)
    plan = load(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Choose a new scoring report; completed observations are not overwritten')
    scorer = VoiceScorer(args.cache) if args.metric == 'voice' else TextScorer(args.metric, args.cache)
    report = {'plan_sha256': plan_sha, 'metric': scorer.metadata, 'originals_only': args.originals_only,
              'candidate_outputs_read': not args.originals_only, 'clips': [], 'complete': False}
    save(args.output, report)
    for item in plan['items']:
        source = ROOT/item['source']
        if digest(args.plan) != plan_sha or digest(source) != item['sha256']:
            raise ValueError('Scoring plan or source changed')
        output = None if args.originals_only or not item.get('output') else ROOT/item['output']
        row = {'id': item['id'], 'source_sha256': item['sha256'], 'language': item['language']}
        if not args.originals_only and (output is None or not output.is_file()):
            row['status'] = 'no_processed_output'
        else:
            before_sha = digest(output) if output else None
            row.update(scorer.score(source, output, item['text'], item['language']), output_sha256=before_sha,
                       status='original_calibration' if args.originals_only else 'scored')
            if output is not None and digest(output) != before_sha:
                raise ValueError('Scored output changed')
        if digest(source) != item['sha256']:
            raise ValueError('Source changed during scoring')
        report['clips'].append(row)
        save(args.output, report)
        print(args.metric, row['id'], row['status'], flush=True)
    report.update(complete=True, completed_at=datetime.now(timezone.utc).isoformat())
    save(args.output, report)


if __name__ == '__main__':
    main()
