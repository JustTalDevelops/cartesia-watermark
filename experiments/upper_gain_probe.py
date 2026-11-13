"""Measure targeted native spectral attenuation after the trained editor."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,load_model,process_audio,quality_metrics
from eval_remover import speech_metrics


def main():
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    checkpoint=ROOT/'results/remover_wave8oracle.pt';model,_=load_model(checkpoint)
    out=ROOT/'audio/upper_gain_probe';out.mkdir(exist_ok=True)
    report=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),clips=[])
    for name in ('ja_08_marked','es_03_marked','ar_00_marked','en_02_marked','fr_04_marked','zh_16_marked'):
        source=ROOT/'audio/remover_validation'/f'{name}.wav';x,sr=sf.read(source);base=process_audio(model,x,sr)
        f=np.fft.rfftfreq(len(base),1/sr)
        for low,high in ((4000,8000),(4000,6000)):
            mask=np.clip((f-low+200)/200,0,1)*np.clip((high+200-f)/200,0,1)
            band=np.fft.irfft(np.fft.rfft(base)*mask,n=len(base))
            for db in (6,12,18,24):
                y=base+(10**(-db/20)-1)*band
                variant=f'band{low}-{high}_minus{db}db'
                target=out/f'{name}_{variant}.wav';sf.write(target,y,sr,subtype='FLOAT')
                row=dict(source=str(source.relative_to(ROOT)),variant=variant,output=str(target.relative_to(ROOT)),
                         **quality_metrics(x,y),**speech_metrics(x,y,sr))
                report['clips'].append(row);print(name,variant,round(row['pesq_wb'],3),round(row['stoi'],4),round(row['estoi'],4),flush=True)
    (ROOT/'results/upper_gain_probe.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
