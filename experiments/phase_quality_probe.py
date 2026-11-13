"""Development-only all-pass, phase modulation and pitch reconstruction probes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT, quality_metrics
from eval_remover import speech_metrics


def variants(x, sr, source):
    analytic = signal.hilbert(x)
    t = np.arange(len(x))/sr
    for rate in (4, 20, 80):
        for angle in (.5, 1.5):
            yield f'phasemod{rate}_{angle}', np.real(analytic*np.exp(1j*angle*np.sin(2*np.pi*rate*t)))
    for r in (.5, .8, .95):
        stages = []
        for freq in np.arange(700, 6000, 250):
            c = -2*r*np.cos(2*np.pi*freq/sr)
            stages.append([r*r,c,1.,1.,c,r*r])
        y = signal.sosfilt(np.array(stages), x)
        # Compensate a single global delay; never realign local speech segments.
        corr = signal.correlate(y, x, mode='full', method='fft')
        lags = signal.correlation_lags(len(y),len(x))
        legal = np.abs(lags) <= sr//50
        lag = lags[legal][np.argmax(corr[legal])]
        if lag > 0:
            y = np.pad(y[lag:], (0,lag))
        elif lag < 0:
            y = np.pad(y[:lag], (-lag,0))
        yield f'allpass{r}', y
    for pitch in (1.01, 1.025, 1.05):
        result = subprocess.run(['ffmpeg','-v','error','-i',str(source),'-af',
            f'rubberband=pitch={pitch}:formant=preserved','-f','f32le','-'],capture_output=True,check=True)
        y = np.frombuffer(result.stdout, np.float32).astype(float)
        y = np.pad(y[:len(x)], (0,max(0,len(x)-len(y))))
        yield f'pitch{pitch}', y


def main():
    output = ROOT/'audio/phase_quality_probe'; output.mkdir(exist_ok=True)
    report = []
    for name in ('es_228_sonic-preview','en_118_sonic-preview','ja_335_sonic-3.6','pt_346_sonic-3.6'):
        source = ROOT/'audio/corpus5'/f'{name}.wav'
        x,sr = sf.read(source)
        for variant,y in variants(x,sr,source):
            # A global gain avoids playback clipping without introducing a limiter.
            peak = np.abs(y).max()
            if peak > .999:
                y = y*(.999/peak)
            row = dict(source=str(source.relative_to(ROOT)),variant=variant,
                       **quality_metrics(x,y), **speech_metrics(x,y,sr))
            if row['pesq_wb'] >= 4.0 and row['estoi'] >= .975:
                target = output/f'{name}_{variant}.wav'
                sf.write(target,y,sr,subtype='FLOAT')
                row.update(output=str(target.relative_to(ROOT)),output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            report.append(row)
            print(name[:2],variant,round(row['pesq_wb'],3),round(row['estoi'],4),flush=True)
    (ROOT/'results/phase_quality_probe.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
