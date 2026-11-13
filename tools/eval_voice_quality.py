"""Paired DNSMOS P.835 estimates; independent of remover training losses.

    python tools/eval_voice_quality.py results/remover_quality_v2_testbatch2_h2h4.json \
        --output results/remover_quality_v2_dnsmos.json

Uses Microsoft's published waveform model, windowing, and regular-MOS
calibration: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
Model/license: Microsoft DNS Challenge, CC BY 4.0. The pinned, unmodified model
and license are downloaded into .cache. Short recordings are repeated only
inside this quality estimator, never for watermark endpoint verification.
These are model estimates, not listening-test scores or watermark evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import requests
import soundfile as sf

from remover import ROOT

REVISION = '785ab12275d5cf9341c5e946b0eee9f8825ece3b'
MODEL_SHA256 = '269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd'
LICENSE_SHA256 = 'd6239afa918961b465b07bf7411cbe34ff6685854f58553db7966f4881a0211f'
RATE, WINDOW = 16000, 144160


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VoiceQuality:
    def __init__(self):
        directory = ROOT/'.cache/dnsmos'
        directory.mkdir(parents=True, exist_ok=True)
        for source, name, expected in (
            ('DNSMOS/DNSMOS/sig_bak_ovr.onnx', 'sig_bak_ovr.onnx', MODEL_SHA256),
            ('LICENSE', 'LICENSE', LICENSE_SHA256),
        ):
            path = directory/name
            if not path.exists():
                url = f'https://raw.githubusercontent.com/microsoft/DNS-Challenge/{REVISION}/{source}'
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                if hashlib.sha256(response.content).hexdigest() != expected:
                    raise ValueError('Downloaded DNSMOS artifact hash mismatch')
                path.write_bytes(response.content)
            if digest(path) != expected:
                raise ValueError('Cached DNSMOS artifact hash mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory/'sig_bak_ovr.onnx'), options,
                                           providers=['CPUExecutionProvider'])
        self.metadata = {'implementation': 'dnsmos-p835-regular-v1', 'revision': REVISION,
                         'model_sha256': MODEL_SHA256, 'librosa': librosa.__version__,
                         'onnxruntime': ort.__version__, 'resampler': 'librosa-soxr_hq',
                         'sample_rate': RATE, 'window_samples': WINDOW, 'hop_samples': RATE,
                         'short_clip_policy': 'repeat inside quality estimator only'}

    def score(self, path):
        audio, sample_rate = sf.read(path)
        audio = audio.mean(1) if audio.ndim == 2 else audio
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Quality scoring requires finite nonempty audio')
        if sample_rate != RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=RATE, res_type='soxr_hq')
        natural_length = len(audio)
        while len(audio) < WINDOW:
            audio = np.concatenate((audio, audio))
        hops = int(np.floor(len(audio)/RATE)-9.01)+1
        scores = []
        for index in range(hops):
            segment = audio[index*RATE:index*RATE+WINDOW].astype(np.float32)[None]
            signal, background, overall = self.session.run(None, {'input_1': segment})[0][0]
            scores.append([
                np.polyval([-.08397278, 1.22083953, .0052439], signal),
                np.polyval([-.13166888, 1.60915514, -.39604546], background),
                np.polyval([-.06766283, 1.11546468, .04602535], overall),
            ])
        if not scores or not np.isfinite(scores).all():
            raise ValueError('Invalid DNSMOS result')
        mean = np.mean(scores, axis=0)
        return dict(zip(('signal', 'background', 'overall'), map(float, mean)),
                    windows=len(scores), natural_seconds=natural_length/RATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+', help='Existing paired remover evaluation reports')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    scorer = VoiceQuality()
    cache_path = ROOT/'.cache/dnsmos/quality_scores.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {'metadata': scorer.metadata, 'scores': {}}
    if cache['metadata'] != scorer.metadata:
        raise ValueError('Quality cache uses a different model or processing version')

    def score(path, expected):
        path = ROOT/path
        sha = digest(path)
        if sha != expected:
            raise ValueError('Paired audio differs from its evaluation report')
        if sha not in cache['scores']:
            cache['scores'][sha] = scorer.score(path)
            temporary = cache_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(cache, indent=2))
            temporary.replace(cache_path)
        return cache['scores'][sha]

    comparisons = []
    for path in args.reports:
        data = json.loads(path.read_text())
        rows = []
        for row in data['clips']:
            before = score(row['source'], row['source_sha256'])
            after = score(row['output'], row['output_sha256'])
            rows.append({'source': row['source'], 'output': row['output'], 'marked': row['marked'],
                         'before': before, 'after': after,
                         'delta': {key: after[key]-before[key] for key in ('signal', 'background', 'overall')}})
        marked = [r for r in rows if r['marked']]
        summary = {key: {'median_before': float(np.median([r['before'][key] for r in marked])),
                         'median_after': float(np.median([r['after'][key] for r in marked])),
                         'median_paired_delta': float(np.median([r['delta'][key] for r in marked])),
                         'minimum_paired_delta': float(min(r['delta'][key] for r in marked))}
                   for key in ('signal', 'background', 'overall')}
        comparisons.append({'report': str(path), 'report_sha256': digest(path),
                            'marked_clips': len(marked), 'summary': summary, 'clips': rows})
        print(path.name, json.dumps(summary), flush=True)
    args.output.write_text(json.dumps({'metric': scorer.metadata, 'comparisons': comparisons,
                                      'limitation': 'Predicted quality, not a human listening test; 16kHz input. Native-band fidelity must also be checked.'}, indent=2))


if __name__ == '__main__':
    main()
