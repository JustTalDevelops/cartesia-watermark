"""Measure physical baselines before spending endpoint queries on them.

Only development clips; no sealed4 audio. No API calls.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from remover import ROOT, quality_metrics
from eval_remover import speech_metrics


def variants(x, sr):
    for lo, hi in ((1500, 3500), (1000, 4000), (500, 5000)):
        # Zero phase bands keep the unedited component in exact alignment.
        band = signal.sosfiltfilt(signal.butter(6, (lo, hi), 'bandpass', fs=sr, output='sos'), x)
        for db in (-1.5, -3, -6):
            yield f'eq{lo}-{hi}_{abs(db):g}db', x + (10**(db/20)-1)*band
        for ms in (5, 15, 33):
            d = round(sr*ms/1000)
            yield f'delay{lo}-{hi}_{ms}ms', x-band+np.pad(band[:-d], (d, 0))
        q = signal.hilbert(band).imag
        for degrees in (30, 60, 90):
            angle = np.deg2rad(degrees)
            yield f'phase{lo}-{hi}_{degrees}', x + (np.cos(angle)-1)*band-np.sin(angle)*q
    q = signal.hilbert(x).imag
    for degrees in (15, 30, 60, 90):
        a = np.deg2rad(degrees)
        yield f'fullphase{degrees}', np.cos(a)*x-np.sin(a)*q


def main():
    out = ROOT/'audio/signal_quality_probe'
    out.mkdir(exist_ok=True)
    report = []
    for name in ('en_118_sonic-preview', 'es_228_sonic-preview', 'ja_335_sonic-3.6', 'pt_346_sonic-3.6'):
        path = ROOT/'audio/corpus5'/f'{name}.wav'
        x, sr = sf.read(path)
        for variant, y in variants(x, sr):
            metrics = speech_metrics(x, y, sr)
            row = dict(source=str(path.relative_to(ROOT)), variant=variant,
                       **quality_metrics(x, y), **metrics)
            # Save usable candidates; rejected variants still stay in the report.
            if metrics.get('pesq_wb', 0) >= 4.15 and metrics['estoi'] >= .98:
                target = out/f'{name}_{variant}.wav'
                sf.write(target, y, sr, subtype='FLOAT')
                row['output'] = str(target.relative_to(ROOT))
            report.append(row)
            print(name[:2], variant, round(metrics.get('pesq_wb', 0), 3), round(metrics['estoi'], 4), flush=True)
    (ROOT/'results/signal_quality_probe.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
