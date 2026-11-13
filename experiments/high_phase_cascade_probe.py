"""Native inference of the editor followed by learned high-band phase repair."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from remover import ROOT,load_model,process_audio,quality_metrics
from eval_remover import speech_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default='high_phase_cascade')
    parser.add_argument('--blur',type=float,default=2.)
    parser.add_argument('--time-blur',type=float,default=0.)
    parser.add_argument('--sources',nargs='+',default=['es_03_marked','ja_08_marked','ar_00_marked','en_02_marked','fr_04_marked','zh_16_marked'])
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():parser.error('Use an alphanumeric tag')
    torch.set_num_threads(2);torch.set_flush_denormal(True)
    core_path=ROOT/'results/remover_wave8oracle.pt'
    high_path=ROOT/'results/band_rebuilder_highnoise.pt'
    core,_=load_model(core_path);high,_=load_model(high_path)
    high.config.update(phase_hint=True,mix=1.,min_snr_db=None,post_lowpass_hz=8000.,magnitude_blur=args.blur,magnitude_time_blur=args.time_blur)
    out=ROOT/'audio'/args.tag;out.mkdir(exist_ok=True)
    report=dict(core_sha256=hashlib.sha256(core_path.read_bytes()).hexdigest(),high_sha256=hashlib.sha256(high_path.read_bytes()).hexdigest(),high_config=high.config,clips=[])
    for name in args.sources:
        source=ROOT/'audio/remover_validation'/f'{name}.wav';x,sr=sf.read(source)
        y=process_audio(high,process_audio(core,x,sr),sr)
        target=out/source.name;sf.write(target,y,sr,subtype='FLOAT')
        row=dict(source=str(source.relative_to(ROOT)),output=str(target.relative_to(ROOT)),**quality_metrics(x,y),**speech_metrics(x,y,sr))
        report['clips'].append(row);print(name,round(row['pesq_wb'],3),round(row['stoi'],4),round(row['estoi'],4),flush=True)
    (ROOT/f'results/{args.tag}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
