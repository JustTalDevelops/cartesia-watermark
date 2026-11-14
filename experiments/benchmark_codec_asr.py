"""Exploratory FP32/BF16/INT8 recognition benchmark on spent recordings only.

Does not modify the processing runtime, select audio, call APIs or read fresh
study outputs. MMS is loaded anew per language: adapter switching after eager
quantization is not supported by this experiment. Timings exclude model loading
and conversion; report all repetitions and transcript differences honestly.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.score_codec_study import TextScorer, digest
from speech_repair import disagreement, units


def weight_bytes(model, torch):
    floating = sum(p.numel()*p.element_size() for p in model.parameters())
    quantized = sum(m.weight().numel()*m.weight().element_size() for m in model.modules()
                    if isinstance(m, torch.ao.nn.quantized.dynamic.Linear))
    return floating+quantized


def benchmark(plan_path, directory, repetitions=2, methods=('fp32', 'dynamic-int8-linear')):
    import torch
    plan = json.loads(plan_path.read_text())
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory/'result.json'
    report = {'plan_sha256': digest(plan_path), 'source_sha256': digest(Path(__file__)),
              'threads': 4, 'engine': torch.backends.quantized.engine,
              'methods': list(methods), 'repetitions': repetitions,
              'fresh_sources_read': 0, 'api_calls': 0, 'runtime_modified': False,
              'scope': 'Two spent generated sources, their selected development outputs, exploratory timings only; MMS quantization is fixed-language, not a deployed multilingual runtime.',
              'torch': torch.__version__, 'cases': [], 'complete': False}

    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(report_path)

    save()
    for metric in ('turbo', 'mms'):
        for language in sorted({i['language'] for i in plan['items']}):
            items = [i for i in plan['items'] if i['language'] == language]
            scorer = TextScorer(metric, directory/'unused-cache')
            if metric == 'mms':
                from codec_settings import MMS_CODES
                adapter = MMS_CODES[language]
                scorer.processor.tokenizer.set_target_lang(adapter)
                scorer.model.load_adapter(adapter, local_files_only=True, use_safetensors=True)
                scorer.model.eval()
                scorer.language = adapter
            original_units = {}
            for method in methods:
                conversion = 0
                if method.startswith('dynamic-int8'):
                    names = {name for name, module in scorer.model.named_modules()
                             if isinstance(module, torch.nn.Linear) and name != 'proj_out'}
                    started = time.perf_counter()
                    configuration = ({name: torch.ao.quantization.per_channel_dynamic_qconfig for name in names}
                                     if method == 'dynamic-int8-channel' else names)
                    torch.ao.quantization.quantize_dynamic(scorer.model, configuration, dtype=torch.qint8, inplace=True)
                    scorer.model.eval()
                    conversion = time.perf_counter()-started
                size = weight_bytes(scorer.model, torch)
                for item in items:
                    path = ROOT/item['path']
                    if digest(path) != item['sha256'] or digest(plan_path) != report['plan_sha256']:
                        raise ValueError('Benchmark input changed')
                    row = {'metric': metric, 'language': language, 'id': item['id'],
                           'input_sha256': item['sha256'], 'method': method, 'times_seconds': [],
                           'weight_tensor_bytes_approximate': size, 'conversion_seconds': conversion,
                           'normalized_units_identical_to_fp32': [], 'normalized_disagreement_from_fp32': []}
                    sequences = []
                    for repetition in range(repetitions):
                        # Each repeat uses an empty cache; never time a cached transcript.
                        scorer.cache = directory/'cache'/metric/language/method/str(repetition)
                        scorer.cache.mkdir(parents=True, exist_ok=True)
                        started = time.perf_counter()
                        context = torch.autocast('cpu', dtype=torch.bfloat16) if method == 'bf16-autocast' else nullcontext()
                        with context:
                            text = scorer.transcribe(path, language)
                        row['times_seconds'].append(time.perf_counter()-started)
                        sequence = units(text, scorer.normalize, language)
                        sequences.append(sequence)
                        if method == 'fp32' and repetition == 0:
                            original_units[item['id']] = sequence
                        before = original_units[item['id']]
                        row['normalized_units_identical_to_fp32'].append(before == sequence)
                        row['normalized_disagreement_from_fp32'].append(disagreement(before, sequence))
                    row['deterministic_across_repetitions'] = all(s == sequences[0] for s in sequences)
                    if digest(path) != item['sha256']:
                        raise ValueError('Benchmark input changed during inference')
                    report['cases'].append(row)
                    save()
                    print(metric, language, method, item['id'], row['times_seconds'],
                          row['normalized_units_identical_to_fp32'], flush=True)
            del scorer
            gc.collect()
    report['complete'] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=('fp32', 'bf16-autocast', 'dynamic-int8-linear', 'dynamic-int8-channel'),
                        default=['fp32', 'dynamic-int8-linear'])
    args = parser.parse_args()
    if args.methods[0] != 'fp32' or len(set(args.methods)) != len(args.methods):
        parser.error('Methods must start with a unique FP32 baseline')
    if sum(m.startswith('dynamic-int8') for m in args.methods) > 1 or any(
            m.startswith('dynamic-int8') for m in args.methods[:-1]):
        parser.error('Choose at most one INT8 method, last; conversion mutates the benchmark model')
    benchmark(args.plan, args.directory, methods=args.methods)


if __name__ == '__main__':
    main()
