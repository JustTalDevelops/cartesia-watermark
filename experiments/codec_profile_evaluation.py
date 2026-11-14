"""Prospective reference scoring, separate from reconstruction and selection.

Hindi uses Turbo evaluation; Arabic uses Large V3 plus MMS; other languages use
Small. All use the prospective processing text policy. Raw word diagnostics are
also retained. This does not rewrite the original frozen codec study.
"""
import argparse
import gc
import hashlib
import json
from math import isfinite
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from codec_inference import check_environment, verify_files
from codec_text import paired_error, text_units
from codec_content import secondary_for
from codec_settings import MMS_CODES
from experiments.score_codec_study import TextScorer, digest, save


def evaluation_metrics(language):
    return ('turbo',) if language == 'hi' else ('large', 'mms') if language == 'ar' else ('small',)


def limits(language):
    return (.01, .10) if language in ('hi', 'ta') else (.03, .15)


def quality_criteria(language, paired, guard_deltas, evaluation_deltas, original_errors,
                     speaker_minimum, dnsmos_delta, content_guard_passed, language_matches_request):
    """Frozen qualification arithmetic; missing or failed evidence never passes."""
    if language not in MMS_CODES:
        raise ValueError('Qualification requires a supported language profile')
    primary, secondary = ('large' if language in ('hi', 'ta') else 'turbo'), secondary_for(language)
    guards, evaluations = (primary, secondary), evaluation_metrics(language)
    if (set(paired) != set(guards) or set(guard_deltas) != set(guards) or
            set(evaluation_deltas) != set(evaluations) or set(original_errors) != set(guards) | set(evaluations)):
        raise ValueError('Every declared guard/evaluation checkpoint must remain in qualification')
    maximum, ceiling = limits(language)
    finite = lambda value: type(value) in (int, float) and isfinite(value)
    passed = lambda value: finite(value) and value <= maximum+1e-12
    gates = {name+'_paired': finite(paired[name]) and 0 <= paired[name] <= maximum+1e-12 for name in guards}
    gates.update({name+'_reference_delta': passed(guard_deltas[name]) for name in guards})
    gates.update({'evaluation_'+name+'_reference_delta': passed(evaluation_deltas[name]) for name in evaluations})
    gates.update(speaker_minimum=finite(speaker_minimum) and .95 <= speaker_minimum <= 1.,
                 dnsmos_overall=finite(dnsmos_delta) and dnsmos_delta >= -.2)
    calibrated = sum(finite(error) and 0 <= error <= ceiling+1e-12 for error in original_errors.values())
    source_confident = language_matches_request is True and calibrated >= 2
    return {'quality_gate_results': gates, 'calibrated_original_checkpoints': calibrated,
            'source_calibration_sufficient': source_confident,
            'all_estimated_quality_gates_passed': content_guard_passed is True and all(gates.values()),
            'quality_qualified': source_confident and content_guard_passed is True and all(gates.values())}


def reference_score(reference, recognized, language, normalize):
    truth, predicted = text_units(reference, normalize, language), text_units(recognized, normalize, language)
    error = paired_error(truth, predicted, language)
    # A diagnostic word view retains essential marks and the original script.
    # It never replaces the declared character/spacing-normalized criteria.
    def raw_words(value):
        value = unicodedata.normalize('NFKC', value).lower()
        return ''.join(' ' if unicodedata.category(c)[0] in 'SP' else c for c in value).split()
    raw_error = paired_error(raw_words(reference), raw_words(recognized), None)
    maximum, ceiling = limits(language)
    return {'reference_error': error, 'reference_units': len(truth), 'recognized_units': len(predicted),
            'comparison_units': 'characters' if language in ('hi', 'ta', 'ja', 'zh') else
                                'KsponSpeech spacing-normalized words' if language == 'ko' else 'words',
            'raw_word_reference_error_diagnostic_only': raw_error,
            'maximum_reference_delta': maximum, 'original_calibration_ceiling': ceiling,
            'calibrated': error is not None and error <= ceiling+1e-12}


class LargeTextScorer(TextScorer):
    def __init__(self, cache):
        check_environment()
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        torch.set_num_threads(4)
        entry = json.loads((ROOT/'tools/codec_assets.json').read_text())['large']
        folder = ROOT/entry['local_directory']
        verify_files(folder, entry['files_sha256'])
        self.metric, self.cache, self.language = 'large', cache, None
        self.normalize = BasicTextNormalizer(remove_diacritics=False)
        self.processor = WhisperProcessor.from_pretrained(folder, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            folder, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').eval().requires_grad_(False)
        self.model.generation_config.forced_decoder_ids = None
        self.metadata = {'model': entry['model_id'], 'revision': entry['revision'], 'license': entry['license'],
                         'attention': 'sdpa', 'precision': 'float32',
                         'decode': 'Native timestamps, minimum30s features, complete longer input, max_new_tokens440, no sampling/previous-text conditioning',
                         'role': 'Arabic evaluation transcription; shares Whisper ancestry with Turbo',
                         'reference_normalization': 'Raw transcription cache; scoring uses the prospective codec_text policy'}
        self.configuration = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()


def score(plan_path, output_path, cache, originals_only):
    plan_sha = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    cache.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError('Scoring reports are not silently overwritten')
    report = {'plan_sha256': plan_sha, 'originals_only': originals_only,
              'candidate_outputs_read': not originals_only, 'checkpoints': {}, 'clips': [], 'complete': False,
              'roles': {'hi': 'Turbo evaluation transcription; the checkpoint also detects processing language',
                        'ar': 'Large and MMS evaluation; Turbo/FastConformer perform processing guards',
                        'other': 'Small evaluation; Whisper checkpoints share ancestry'},
              'normalization': 'Prospective codec_text policy; vowel-preserving Hindi/Tamil characters at one percent; pinned Chinese conversion and Korean spacing'}
    save(output_path, report)
    for metric in ('small', 'turbo', 'large', 'mms'):
        selected = [i for i in plan['items'] if metric in evaluation_metrics(i['language'])]
        if not selected:
            continue
        scorer = LargeTextScorer(cache) if metric == 'large' else TextScorer(metric, cache)
        report['checkpoints'][metric] = {'raw_transcription_helper_metadata': scorer.metadata,
                                         'study_role': 'Evaluation transcription only; see the language roles'}
        for item in selected:
            source = ROOT/item['source']
            if digest(plan_path) != plan_sha or digest(source) != item['sha256']:
                raise ValueError('Plan or complete source changed')
            before = reference_score(item['text'], scorer.transcribe(source, item['language']),
                                     item['language'], scorer.normalize)
            row = {'id': item['id'], 'metric': metric, 'language': item['language'], 'source_sha256': item['sha256'],
                   'original': before, 'status': 'original_calibration' if originals_only else 'no_processed_output'}
            candidate = ROOT/item['output'] if item.get('output') else None
            if not originals_only and candidate is not None and candidate.is_file():
                checksum = digest(candidate)
                after = reference_score(item['text'], scorer.transcribe(candidate, item['language']),
                                        item['language'], scorer.normalize)
                delta = (None if before['reference_error'] is None or after['reference_error'] is None else
                         after['reference_error']-before['reference_error'])
                row.update(output_sha256=checksum, processed=after, reference_error_delta=delta,
                           reference_delta_passed=delta is not None and delta <= limits(item['language'])[0]+1e-12,
                           status='scored')
                if digest(candidate) != checksum:
                    raise ValueError('Evaluated candidate changed')
            if digest(source) != item['sha256']:
                raise ValueError('Source changed during evaluation')
            report['clips'].append(row)
            save(output_path, report)
            print('PROFILE_EVALUATION', metric, item['id'], row['status'], flush=True)
        del scorer
        gc.collect()
    report['complete'] = True
    save(output_path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--originals-only', action='store_true')
    args = parser.parse_args()
    score(args.plan, args.output, args.cache, args.originals_only)
