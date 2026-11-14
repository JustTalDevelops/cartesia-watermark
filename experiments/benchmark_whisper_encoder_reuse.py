"""Compare exact language/token results and FP32 encoder calls on spent files."""
import argparse
from functools import wraps
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.score_codec_study import TextScorer, digest
from experiments.whisper_encoder_reuse import prepare_original


def benchmark(plan_path, directory):
    import torch
    from math import gcd
    import soundfile as sf
    from scipy.signal import resample_poly
    directory.mkdir(parents=True, exist_ok=False)
    plan = json.loads(plan_path.read_text())
    scorer = TextScorer('turbo', directory/'unused')
    model = scorer.model
    encoder = model.get_encoder()
    original_forward = encoder.forward
    counter = []

    @wraps(original_forward)
    def counted(*args, **kwargs):
        counter.append(1)
        return original_forward(*args, **kwargs)

    encoder.forward = counted
    report = {'plan_sha256': digest(plan_path), 'benchmark_source_sha256': digest(Path(__file__)),
              'helper_source_sha256': digest(ROOT/'experiments/whisper_encoder_reuse.py'),
              'torch': torch.__version__, 'precision': 'float32', 'threads': 4,
              'model': scorer.metadata, 'fresh_sources_read': 0, 'api_calls': 0,
              'frozen_runtime_modified': False, 'cases': [], 'complete': False}
    output = directory/'result.json'
    for item in plan['items']:
        source = ROOT/item['path']
        if digest(source) != item['sha256']:
            raise ValueError('Benchmark input changed')
        audio, rate = sf.read(source, dtype='float32')
        audio = audio.mean(1) if audio.ndim == 2 else audio
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000//divisor, rate//divisor)
        batch = scorer.processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True,
                                 truncation=False, padding='max_length', max_length=max(len(audio), 30*16000))
        baseline = None
        for method in ('original', 'encoder-reuse'):
            row = {'id': item['id'], 'input_sha256': item['sha256'], 'method': method,
                   'times_seconds': [], 'encoder_forward_calls': [], 'exact_token_parity': [],
                   'language_parity': [], 'encoder_reused': []}
            for repetition in range(2):
                before = len(counter)
                started = time.perf_counter()
                with torch.inference_mode():
                    if item['id'].endswith('_original'):
                        language, parameters, reused = prepare_original(model, batch, reuse=method == 'encoder-reuse')
                    else:
                        language, parameters, reused = item['language'], dict(batch), False
                    prediction = model.generate(**parameters, language=language, task='transcribe', return_timestamps=True,
                                                max_new_tokens=440, do_sample=False, condition_on_prev_tokens=False)
                row['times_seconds'].append(time.perf_counter()-started)
                if baseline is None:
                    baseline = (language, prediction.clone())
                row['language_parity'].append(language == baseline[0])
                row['exact_token_parity'].append(torch.equal(prediction, baseline[1]))
                row['encoder_forward_calls'].append(len(counter)-before)
                row['encoder_reused'].append(reused)
            if digest(source) != item['sha256'] or digest(plan_path) != report['plan_sha256']:
                raise ValueError('Benchmark input changed during inference')
            report['cases'].append(row)
            output.write_text(json.dumps(report, indent=2)+'\n')
            print(item['id'], method, row, flush=True)
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.plan, args.directory)
