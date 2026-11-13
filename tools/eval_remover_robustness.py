"""Check local suppression after ordinary audio encoding, without API calls.

Encoding is applied to detector inputs only. The remover's returned WAVs are
unchanged. A restored local positive can expose fragile detector suppression;
it does not independently establish watermark presence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ['FEAT'] = 'xlsr2'
import numpy as np
import soundfile as sf
import torch

import frames
import train_head as th
from final_detector import Detector
from remover import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--languages', nargs='+', default=['ar', 'en', 'es', 'fr', 'ja', 'zh'])
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Threads must be positive')
    if not args.tag.replace('_', '').replace('-', '').isalnum():
        parser.error('Use an alphanumeric tag')
    output = ROOT / f'results/remover_{args.tag}_robustness.json'
    if output.exists():
        parser.error('Report exists; use a new tag')
    torch.set_num_threads(args.threads)
    detectors = {name: Detector(config=ROOT / f'results/{config}.json') for name, config in
                 [('r4', 'final_detector_r4'), ('r1', 'oracle_detector_r1')]}

    @torch.no_grad()
    def scores(audio, rate):
        mono = audio.mean(1) if audio.ndim == 2 else audio
        hidden, rms = frames.frames(frames.to16k(mono, rate))
        windows = th.windows_of(hidden, rms)
        if not windows:
            return {name: None for name in detectors}
        features = torch.tensor(np.array(windows)).float()
        return {name: float(np.mean([float(head(features).mean()) for head in detector.frozen]))
                for name, detector in detectors.items()}

    report = json.loads(args.report.read_text())
    rows = []
    for row in report['clips']:
        if not row['marked'] or Path(row['source']).name.split('_')[0] not in args.languages:
            continue
        item = {'source': row['source'], 'output': row['output']}
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for role, key in [('original', 'source'), ('edited', 'output')]:
                path = ROOT / row[key]
                if hashlib.sha256(path.read_bytes()).hexdigest() != row[key + '_sha256']:
                    raise ValueError('Evaluation audio changed')
                audio, rate = sf.read(path)
                variants = {'native': scores(audio, rate)}
                sf.write(temp / 'pcm16.wav', audio, rate, subtype='PCM_16')
                encoded, encoded_rate = sf.read(temp / 'pcm16.wav')
                variants['pcm16'] = scores(encoded, encoded_rate)
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path),
                                '-b:a', '128k', str(temp / 'audio.mp3')], check=True)
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(temp / 'audio.mp3'),
                                '-c:a', 'pcm_f32le', str(temp / 'decoded.wav')], check=True)
                encoded, encoded_rate = sf.read(temp / 'decoded.wav')
                variants['mp3_128k'] = scores(encoded, encoded_rate)
                item[role] = variants
        rows.append(item)
        print(Path(row['source']).name, json.dumps(item['edited']), flush=True)
    summary = {}
    for name, detector in detectors.items():
        threshold = detector.threshold
        counts = {variant: {
            'original_positive': sum(r['original'][variant][name] is not None and r['original'][variant][name] > threshold for r in rows),
            'edited_positive': sum(r['edited'][variant][name] is not None and r['edited'][variant][name] > threshold for r in rows)}
            for variant in ('native', 'pcm16', 'mp3_128k')}
        counts['native_negative_restored_by_mp3'] = sum(
            r['edited']['native'][name] is not None and r['edited']['mp3_128k'][name] is not None and
            r['edited']['native'][name] <= threshold < r['edited']['mp3_128k'][name] for r in rows)
        summary[name] = {'threshold': threshold, **counts}
    output.write_text(json.dumps({'source_report': str(args.report),
        'source_report_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
        'used_for_model_selection': True, 'api_queries': 0, 'summary': summary, 'clips': rows}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
