"""Use clean-band predicted phase with a speech-magnitude fidelity hint.

The learned phase predictor cannot see the source's marked band. Frequency
smoothing optionally removes fine structure from the source magnitude hint.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,load_model,resample,quality_metrics
from eval_remover import speech_metrics


def main():
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    checkpoint=ROOT/'results/band_rebuilder_r1.pt'
    model,_=load_model(checkpoint)
    output=ROOT/'audio/hybrid_band_probe';output.mkdir(exist_ok=True)
    rows=[]
    with torch.no_grad():
        for source in ('audio/corpus5/es_228_sonic-preview.wav','audio/remover_validation/es_03_marked.wav','audio/remover_validation/ja_08_marked.wav'):
            x,sr=sf.read(ROOT/source);x16=torch.tensor(resample(x,sr,16000),dtype=torch.float32)[None]
            clean=model(x16)
            for size in (512,1024):
                w=torch.hann_window(size)
                original=torch.stft(x16,size,size//4,window=w,return_complex=True)
                predicted=torch.stft(clean,size,size//4,window=w,return_complex=True)
                freq=torch.fft.rfftfreq(size,1/16000)
                mask=((freq-1400)/100).clamp(0,1)*((3600-freq)/100).clamp(0,1)
                for blur in (0,2,5):
                    magnitude=original.abs()
                    if blur:magnitude=torch.from_numpy(gaussian_filter1d(magnitude.numpy(),blur,axis=1))
                    z=original*(1-mask[None,:,None])+torch.polar(magnitude,predicted.angle())*mask[None,:,None]
                    rebuilt=torch.istft(z,size,size//4,window=w,length=x16.shape[-1])
                    delta=resample((rebuilt-x16)[0].numpy(),16000,sr)[:len(x)]
                    for mix in (.5,1.):
                        y=x+mix*delta
                        row=dict(source=source,size=size,blur=blur,mix=mix,**quality_metrics(x,y),**speech_metrics(x,y,sr))
                        if row['pesq_wb']>=4.2 and row['stoi']>=.98:
                            dest=output/f'{Path(source).stem}_{size}_blur{blur}_mix{mix}.wav';sf.write(dest,y,sr,subtype='FLOAT');row['output']=str(dest.relative_to(ROOT))
                        rows.append(row);print(Path(source).stem,size,blur,mix,round(row['pesq_wb'],3),round(row['stoi'],4),round(row['estoi'],4),flush=True)
    (ROOT/'results/hybrid_band_probe.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),clips=rows),indent=2))


if __name__=='__main__':main()
