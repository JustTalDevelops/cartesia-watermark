"""Quality of bounded temporal gain correction across the entire signal."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,load_model,process_audio,quality_metrics
from eval_remover import speech_metrics


def main():
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    checkpoint=ROOT/'results/remover_wave8oracle.pt';model,_=load_model(checkpoint)
    out=ROOT/'audio/envelope_probe';out.mkdir(exist_ok=True)
    report=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),clips=[])
    for name in ('ja_08_marked','es_03_marked','ar_00_marked','en_02_marked','fr_04_marked','zh_16_marked'):
        source=ROOT/'audio/remover_validation'/f'{name}.wav';x,sr=sf.read(source);base=process_audio(model,x,sr)
        for fast in (5,10,20):
            slow=80
            a=uniform_filter1d(base*base,max(1,round(fast*sr/1000)))
            b=uniform_filter1d(base*base,round(slow*sr/1000))
            ratio=np.sqrt(np.maximum(b,1e-14)/np.maximum(a,1e-14))
            for exponent in (.5,1.):
                for cap in (1.5,3.):
                    gain=np.clip(ratio**exponent,10**(-cap/20),10**(cap/20))
                    gain=uniform_filter1d(gain,max(1,round(.002*sr)))
                    y=base*gain
                    if np.abs(y).max()>1:y*=.999/np.abs(y).max()
                    variant=f'fast{fast}_power{exponent}_cap{cap}'
                    row=dict(source=str(source.relative_to(ROOT)),variant=variant,fast_ms=fast,slow_ms=slow,exponent=exponent,cap_db=cap,
                             **quality_metrics(x,y),**speech_metrics(x,y,sr))
                    if row['pesq_wb']>=4.2 and row['stoi']>=.98:
                        target=out/f'{name}_{variant}.wav';sf.write(target,y,sr,subtype='FLOAT');row['output']=str(target.relative_to(ROOT))
                    report['clips'].append(row);print(name,variant,round(row['pesq_wb'],3),round(row['stoi'],4),round(row['estoi'],4),flush=True)
    (ROOT/'results/envelope_probe.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
