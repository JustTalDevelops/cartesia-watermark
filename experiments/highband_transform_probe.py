"""Quality of native upper-band edits after the trained low-band remover."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,load_model,process_audio,quality_metrics
from eval_remover import speech_metrics


def variants(x,sr):
    time=np.arange(len(x))/sr
    for low in (3500,4000):
        band=signal.sosfiltfilt(signal.butter(6,(low,12000),'bandpass',fs=sr,output='sos'),x)
        analytic=signal.hilbert(band)
        for hz in (100,200,400):
            yield f'band{low}_shift{hz}',x-band+np.real(analytic*np.exp(2j*np.pi*hz*time))
        for ms in (33,50):
            delay=round(ms*sr/1000)
            yield f'band{low}_delay{ms}',x-band+np.pad(band[:-delay],(delay,0))


def main():
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    checkpoint=ROOT/'results/remover_wave8oracle.pt';model,_=load_model(checkpoint)
    out=ROOT/'audio/highband_transform_probe';out.mkdir(exist_ok=True)
    report=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),clips=[])
    for name in ('ja_08_marked','es_03_marked','ar_00_marked','en_02_marked','fr_04_marked','zh_16_marked'):
        source=ROOT/'audio/remover_validation'/f'{name}.wav';x,sr=sf.read(source);base=process_audio(model,x,sr)
        for variant,y in variants(base,sr):
            row=dict(source=str(source.relative_to(ROOT)),variant=variant,**quality_metrics(x,y),**speech_metrics(x,y,sr))
            if row['pesq_wb']>=4.2 and row['stoi']>=.98:
                target=out/f'{name}_{variant}.wav';sf.write(target,y,sr,subtype='FLOAT');row['output']=str(target.relative_to(ROOT))
            report['clips'].append(row);print(name,variant,round(row['pesq_wb'],3),round(row['stoi'],4),round(row['estoi'],4),flush=True)
    (ROOT/'results/highband_transform_probe.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
