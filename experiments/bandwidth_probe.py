"""Native-rate bandwidth ablations on failures; no API calls or padding."""
import json
from pathlib import Path
import sys

from scipy import signal
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,load_model,process_audio,quality_metrics
from eval_remover import speech_metrics


def main():
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    wave,_=load_model(ROOT/'results/remover_wave8oracle.pt')
    band,_=load_model(ROOT/'results/remover_band600mix075.pt')
    out=ROOT/'audio/bandwidth_probe';out.mkdir(exist_ok=True)
    rows=[]
    for source in ('audio/corpus5/es_228_sonic-preview.wav','audio/remover_validation/es_03_marked.wav','audio/remover_validation/ja_08_marked.wav'):
        x,sr=sf.read(ROOT/source)
        for variant,y in (('original',x),('wave8',process_audio(wave,x,sr)),('band600',process_audio(band,x,sr))):
            for cutoff in (8000,6000,4000):
                filtered=signal.sosfiltfilt(signal.butter(10,cutoff,'lowpass',fs=sr,output='sos'),y)
                row=dict(source=source,variant=variant,cutoff=cutoff,**quality_metrics(x,filtered),**speech_metrics(x,filtered,sr))
                dest=out/f'{Path(source).stem}_{variant}_lp{cutoff}.wav'
                sf.write(dest,filtered,sr,subtype='FLOAT');row['output']=str(dest.relative_to(ROOT))
                row['quality_qualified']=row['pesq_wb']>=4.2 and row['stoi']>=.98
                rows.append(row);print(Path(source).stem,variant,cutoff,round(row['pesq_wb'],3),round(row['stoi'],4),round(row['estoi'],4),flush=True)
    (ROOT/'results/bandwidth_probe.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
