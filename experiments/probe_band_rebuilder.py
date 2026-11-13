"""Measure native-rate quality of the learned missing-band reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from band_rebuilder import BandRebuilder
from remover import ROOT,resample,quality_metrics
from eval_remover import speech_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tag',required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=BandRebuilder(**state['model_config']);model.load_state_dict(state['state_dict']);model.eval()
    out=ROOT/'audio'/args.tag;out.mkdir(exist_ok=True)
    rows=[]
    with torch.no_grad():
        for name in ('es_228_sonic-preview','en_118_sonic-preview','ja_335_sonic-3.6','pt_346_sonic-3.6'):
            source=ROOT/'audio/corpus5'/f'{name}.wav';x,sr=sf.read(source)
            x16=torch.tensor(resample(x,sr,16000),dtype=torch.float32)[None]
            d=(model(x16)-x16)[0].numpy()
            delta=resample(d,16000,sr)[:len(x)]
            for strength in (.25,.5,.75,1.):
                y=x+strength*delta
                row=dict(source=str(source.relative_to(ROOT)),strength=strength,**quality_metrics(x,y),**speech_metrics(x,y,sr))
                if row['pesq_wb']>=4.2 and row['estoi']>=.985:
                    target=out/f'{name}_{strength}.wav';sf.write(target,y,sr,subtype='FLOAT');row['output']=str(target.relative_to(ROOT))
                rows.append(row);print(name[:2],strength,round(row['pesq_wb'],3),round(row['estoi'],4),flush=True)
    (ROOT/f'results/{args.tag}.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),clips=rows),indent=2))


if __name__=='__main__':main()
