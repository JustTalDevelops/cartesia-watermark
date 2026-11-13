"""Development-only subband phase modulation and partial delay quality probes."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import soundfile as sf

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,quality_metrics
from eval_remover import speech_metrics


def variants(x,sr):
    t=np.arange(len(x))/sr
    for lo,hi in ((1500,3500),(1000,4000)):
        band=signal.sosfiltfilt(signal.butter(6,(lo,hi),'bandpass',fs=sr,output='sos'),x)
        analytic=signal.hilbert(band)
        for rate in (4,20,40):
            for angle in (1.5,3.):
                y=x-band+np.real(analytic*np.exp(1j*angle*np.sin(2*np.pi*rate*t)))
                yield f'band{lo}-{hi}_phase{rate}_{angle}',y
        delay=round(.033*sr)
        shifted=np.pad(band[:-delay],(delay,0))
        for wet in (.1,.25,.5):
            yield f'band{lo}-{hi}_delay33_wet{wet}',x+wet*(shifted-band)
    f,_,z=signal.stft(x,sr,nperseg=2048,noverlap=1536)
    mask=(np.clip((f-1300)/200,0,1)*np.clip((3700-f)/200,0,1))[:,None]
    for sigma in (1,3,6):
        mag=gaussian_filter1d(np.abs(z),sigma,axis=0)
        q=z*(1-mask)+mag*np.exp(1j*np.angle(z))*mask
        yield f'magblur{sigma}',signal.istft(q,sr,nperseg=2048,noverlap=1536)[1][:len(x)]


def main():
    out=ROOT/'audio/modulation_quality_probe';out.mkdir(exist_ok=True)
    rows=[]
    for name in ('es_228_sonic-preview','en_118_sonic-preview','ja_335_sonic-3.6','pt_346_sonic-3.6'):
        source=ROOT/'audio/corpus5'/f'{name}.wav';x,sr=sf.read(source)
        for variant,y in variants(x,sr):
            peak=np.abs(y).max()
            if peak>.999:y*=.999/peak
            row=dict(source=str(source.relative_to(ROOT)),variant=variant,**quality_metrics(x,y),**speech_metrics(x,y,sr))
            if row['pesq_wb']>=4.2 and row['estoi']>=.985:
                dest=out/f'{name}_{variant}.wav';sf.write(dest,y,sr,subtype='FLOAT');row['output']=str(dest.relative_to(ROOT))
            rows.append(row);print(name[:2],variant,round(row['pesq_wb'],3),round(row['estoi'],4),flush=True)
    (ROOT/'results/modulation_quality_probe.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
