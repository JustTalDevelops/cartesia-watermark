"""Freeze a language-balanced API test subset before processing test audio.

Uses saved original Cartesia labels on unchanged h2h4 files. No API calls and
no model scoring. Remaining voices stay available as an untouched later batch.
"""
import argparse
import hashlib
import json
from pathlib import Path

from remover import ROOT
import cartesia


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--exclude-plan',type=Path,action='append',default=[])
    parser.add_argument('--extra-languages',nargs='+',default=['es','ja'])
    args=parser.parse_args()
    if args.batch<1:parser.error('Batch number must be positive')
    output=ROOT/f'results/remover_test_batch{args.batch}.plan.json'
    if output.exists():
        raise ValueError('Test selection is already frozen')
    baseline_path=ROOT/'results/h2h4_cartesia.json'
    baseline=json.loads(baseline_path.read_text())
    plan=json.loads((ROOT/'audio/corpus7_plan.json').read_text())
    excluded_voices={r['voice'] for path in args.exclude_plan for r in json.loads(path.read_text())['items']}
    pools={}
    for key,label in sorted(baseline.items()):
        voice=plan[int(Path(key).stem.split('_')[1])][2]
        if key.endswith('_marked.wav') and label and voice not in excluded_voices:
            pools.setdefault(Path(key).name.split('_')[0],[]).append(key)
    selected=[pool[0] for _,pool in sorted(pools.items())]
    for language in args.extra_languages:
        if language not in pools or len(pools[language])<2:
            parser.error(f'No additional unused original-positive voice for {language}')
        selected.extend(pools[language][1:2])
    if len(selected)!=16:
        raise ValueError('Expected 16 marked test voices with language coverage')
    controls=[]
    for language in ('en','es','ja','ar','fr','de'):
        for key in selected:
            clean=key.replace('_marked.wav','_clean.wav')
            if Path(key).name.startswith(language+'_') and baseline[clean] is False:
                controls.append(clean);break
        if len(controls)==4:break
    if len(controls)!=4:raise ValueError('Need four baseline-negative controls')
    items=[]
    for key in selected+controls:
        path=ROOT/'audio'/key
        index=int(path.stem.split('_')[1]);row=plan[index]
        if row[0]!='sealed4' or row[1]!=path.name.split('_')[0]:
            raise ValueError('Test voice plan mismatch')
        items.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            voice=row[2],language=row[1],marked=key.endswith('_marked.wav'),baseline=baseline[key]))
    result=dict(dataset='h2h4',purpose=f'Sealed remover test batch {args.batch}; never model selection',
        selection=f'First unused original-positive voice per language, plus a second voice for {args.extra_languages}; four original-negative paired controls',
        baseline_ledger=str(baseline_path.relative_to(ROOT)),baseline_ledger_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        excluded_plans=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in args.exclude_plan],
        items=items,remaining_marked_paths=[f'audio/{key}' for key in sorted(baseline) if key.endswith('_marked.wav') and key not in selected and plan[int(Path(key).stem.split('_')[1])][2] not in excluded_voices])
    output.write_text(json.dumps(result,indent=2))
    cache=dict(api_version=cartesia.VERSION,purpose='Import prior baseline responses on source files unchanged during remover work',
        clips=[dict(source=r['path'],source_sha256=r['sha256'],original=r['baseline'],original_cached_from=str(baseline_path.relative_to(ROOT))) for r in items])
    (ROOT/f'results/remover_test_batch{args.batch}_baselines_cartesia.json').write_text(json.dumps(cache,indent=2))
    print('Frozen',len(selected),'marked voices,',len(pools),'languages,',len(controls),'controls')


if __name__=='__main__':main()
