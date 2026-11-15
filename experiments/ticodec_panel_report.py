"""Score and curate the complete declared joint panel without changing selection."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from codec_audio import digest
from experiments.check_ticodec_panel import data

FORMATTER_SHA256=digest(Path(__file__))


def load(path):return json.loads(Path(path).read_text())


def identifier(row):return row['id']+'__'+row['method']


def validated(declaration_path,detector_path,builder=None):
    declaration,measured=load(declaration_path),load(detector_path)
    build=builder or (lambda b,s:data(b,s,Path(declaration['band_code_root'])))
    if ({k:v for k,v in declaration.items() if k!='declared_at'}!=
            build(declaration['generation_file'],declaration['completed_study_file'])):
        raise ValueError('Declared joint inventory changed')
    if (measured.get('complete') is not True or measured.get('error_type') or
            measured['declaration_sha256']!=digest(declaration_path)):
        raise ValueError('Complete the entire joint endpoint phase before reference scoring')
    calls=[]
    def pair(row,role):
        for ordinal in (1,2):
            calls.append({'id':row['id'],'role':role,'ordinal':ordinal,'sha256':row['sha256'],'file':row['file'],
                **({'method':row['method']} if role=='candidate' else {})})
    pair(declaration['sources'][0],'opening_positive_control')
    seen={};rows=[]
    for original in declaration['treatments']:
        row=dict(original)
        if row['detector_eligible']:
            checksum=row['sha256'];reused=checksum in seen
            if not reused:
                pair(row,'candidate');observed=measured['observations'][checksum]
                if (type(observed.get('detected')) is not bool or observed['id']!=row['id'] or
                        observed['method']!=row['method']):raise ValueError('Mismatched exact-file observations')
                seen[checksum]=observed
            row.update(detector_observation_group=checksum,detected=seen[checksum]['detected'],
                       identical_file_observations_reused=reused)
        rows.append(row)
    pair(declaration['sources'][-1],'closing_positive_control')
    if (measured['treatments']!=rows or measured['observations']!=seen or
            len(calls)!=len(measured['checks']) or len(calls)!=len(measured['http_attempts']) or
            len(calls)!=declaration['maximum_http_attempts']):
        raise ValueError('Keep every treatment and every actual request exactly once')
    starts=[]
    for planned,check,attempt in zip(calls,measured['checks'],measured['http_attempts']):
        if (any(check[k]!=v or attempt[k]!=v for k,v in planned.items()) or check['http_status']!=200 or
                attempt['http_status']!=200 or type(check.get('detected')) is not bool or attempt.get('error_type')):
            raise ValueError('Incomplete or mismatched service receipts')
        detected=check['detected'];previous=declaration['historical_exact_file_observations'].get(check['sha256'])
        if ((check['role']!='candidate' and detected is not True) or
                (previous is not None and detected!=previous) or
                (check['role']=='candidate' and detected!=seen[check['sha256']]['detected'])):
            raise ValueError('Conflicting reply or failed positive control')
        starts.append(datetime.fromisoformat(attempt['started_at']))
    if (starts[0]<datetime.fromisoformat(declaration['declared_at']) or
            any((b-a).total_seconds()<79.9 for a,b in zip(starts,starts[1:]))):
        raise ValueError('Declaration timing or shared pacing changed')
    return declaration,measured


def evaluation_plan(declaration_path,detector_path):
    declaration,_=validated(declaration_path,detector_path)
    F=Path(declaration['completed_study_file']).parent;old_plan=F/'evaluation.plan.json'
    study=load(declaration['completed_study_file'])
    if digest(old_plan)!=study['endpoint_receipts']['baseline']['plan_sha256']:
        raise ValueError('Authored references differ from the completed study')
    originals={i['id']:i for i in load(old_plan)['items']}
    sources={s['id']:s for s in declaration['sources']};items=[]
    for row in declaration['treatments']:
        if not row['detector_eligible']:continue
        original=originals[row['id']];source=sources[row['id']]
        items.append({'id':identifier(row),'source':source['file'],'sha256':source['sha256'],
            'language':original['language'],'text':original['text'],'output':row['file']})
    return {'declaration_sha256':digest(declaration_path),'detector_sha256':digest(detector_path),
            'evaluation_wrapper_sha256':FORMATTER_SHA256,
            'all_new_eligible_including_detected':True,'prior_qualified_pilot_reused_without_rescoring':True,'items':items}


def score(declaration_path,detector_path,output):
    from experiments.codec_profile_evaluation import score as frozen_score
    from experiments.probe_codec_frames import without_http
    declaration,_=validated(declaration_path,detector_path);plan=evaluation_plan(declaration_path,detector_path)
    plan_path=output.parent/'evaluation.plan.json';cache=output.parent/'evaluation-cache'
    if output.exists() or plan_path.exists() or cache.exists():raise ValueError('Never overwrite existing evaluation work')
    if digest(Path(__file__))!=FORMATTER_SHA256:raise ValueError('Loaded evaluation wrapper changed')
    output.parent.mkdir(parents=True,exist_ok=True);cache.mkdir()
    plan_path.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    seeds={};old_cache=Path(declaration['completed_study_file']).parent/'scores/cache'
    for original in sorted(old_cache.glob('*.json')):
        checksum=digest(original);target=cache/original.name;shutil.copyfile(original,target)
        if digest(original)!=checksum or digest(target)!=checksum:raise ValueError('Original cache changed while copying')
        seeds[original.name]=checksum
    (output.parent/'evaluation-cache-seed.json').write_text(json.dumps(seeds,indent=2)+'\n')
    with without_http():frozen_score(plan_path,output,cache,originals_only=False)
    validated(declaration_path,detector_path)
    if digest(Path(__file__))!=FORMATTER_SHA256:raise ValueError('Evaluation wrapper changed during scoring')


def export(declaration_path,detector_path,evaluation_path):
    from codec_content import assess
    from codec_text import paired_error,text_units
    from experiments.codec_profile_cache import read_recognition
    from experiments.codec_profile_evaluation import evaluation_metrics,quality_criteria
    from experiments.ticodec_pilot_report import export as pilot_export
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    if digest(Path(__file__))!=FORMATTER_SHA256:raise ValueError('Loaded formatter changed')
    evaluation_sha256=digest(evaluation_path)
    declaration,endpoint=validated(declaration_path,detector_path)
    F=Path(declaration['completed_study_file']).parent;D=F.parent
    plan=evaluation_plan(declaration_path,detector_path);plan_path=evaluation_path.parent/'evaluation.plan.json'
    evaluated=load(evaluation_path);original_eval=load(F/'original_evaluation.json');original_voice=load(F/'original_voice.json')
    needed={m for i in plan['items'] for m in evaluation_metrics(i['language'])}
    if (load(plan_path)!=plan or evaluated.get('complete') is not True or evaluated.get('error_type') or
            evaluated.get('originals_only') is not False or evaluated.get('candidate_outputs_read') is not True or
            evaluated['plan_sha256']!=digest(plan_path) or
            evaluated['checkpoints']!={m:original_eval['checkpoints'][m] for m in needed}):
        raise ValueError('Use the complete fixed evaluation plan and original evaluator roles')
    expected={(i['id'],m) for i in plan['items'] for m in evaluation_metrics(i['language'])}
    scores={(c['id'],c['metric']):c for c in evaluated['clips']}
    if len(scores)!=len(evaluated['clips']) or set(scores)!=expected:
        raise ValueError('Every new eligible treatment needs every declared reference checkpoint')
    original_scores={(c['id'],c['metric']):c for c in original_eval['clips']}
    study=load(declaration['completed_study_file']);cases={c['id']:c for c in study['cases']}
    ti=load(declaration['ticodec_generation_file']);band=load(declaration['generation_file'])
    generated={(c['id'],'ticodec-1g4r'):(c,D/'ticodec_development') for c in ti['cases']}
    generated.update({(c['id'],t['method']):(t,Path(declaration['generation_file']).parent) for c in band['cases'] for t in c['treatments']})
    inputs={i['id']:i for i in plan['items']};identity=ti['recognition_configuration_sha256']
    normalizer=BasicTextNormalizer(remove_diacritics=False);pilot=pilot_export(D);rows=[]
    for item,observed in zip(declaration['treatments'],endpoint['treatments']):
        old=cases[item['id']];raw,panel=generated[(item['id'],item['method'])]
        row={k:item[k] for k in ('id','method','sha256','processing_status','endpoint_phase_status')}
        row.update(language=old['language'],source_sha256=old['source_sha256'],source_format=old['source_format'],
            speech_assessment=raw['speech_assessment'],
            runtime_quality_guard=raw.get('runtime_quality_guard'),qualified_spent_transition=False)
        if item['endpoint_phase_status']=='qualified_pilot_reused':
            row.update(prior_pilot_endpoint_sha256=pilot['endpoint_sha256'],prior_pilot_evaluation_sha256=pilot['evaluation_sha256'],
                qualified_spent_transition=pilot['qualified_spent_transition'],new_detector_requests_for_treatment=0)
        elif item['detector_eligible']:
            authored=inputs[identifier(item)]
            before=read_recognition(F/'recognition/originals',old['source_sha256'],identity)
            after=read_recognition(panel/'recognition/processed',item['sha256'],identity)
            speech=assess(before,after)
            if speech!=raw['speech_assessment'] or not speech['content_guard_passed']:
                raise ValueError('Processing assessment differs from its actual cache')
            guards=(speech['primary_guard'],speech['secondary_guard']);truth=text_units(authored['text'],normalizer,old['language'])
            errors={n:paired_error(truth,before[n],old['language']) for n in guards}
            if errors!=old['original_guard_calibration']['original_reference_error']:
                raise ValueError('Original guard reference calibration changed')
            deltas={}
            for name in guards:
                actual=paired_error(truth,after[name],old['language'])
                deltas[name]=None if actual is None or errors[name] is None else actual-errors[name]
            eval_deltas={};measurements=[]
            for metric in evaluation_metrics(old['language']):
                current=scores[(authored['id'],metric)];original=original_scores[(old['id'],metric)]
                if (current['source_sha256']!=old['source_sha256'] or current['output_sha256']!=item['sha256'] or
                        current['original']!=original['original'] or current['status']!='scored'):
                    raise ValueError('Reference measurement refers to another whole file or calibration')
                errors[metric]=original['original']['reference_error'];actual=current['processed']['reference_error']
                delta=None if actual is None or errors[metric] is None else actual-errors[metric]
                if delta!=current['reference_error_delta']:raise ValueError('Reference delta changed')
                eval_deltas[metric]=delta;measurements.append(current)
            receipt_path=panel/'runtime_voice'/old['id']/(item['sha256']+'.json');receipt=load(receipt_path)
            voice=receipt['clips'][0];quality=raw['runtime_quality_guard']
            original=next(c for c in original_voice['clips'] if c['id']==old['id'])
            if (digest(receipt_path)!=quality['receipt_sha256'] or receipt['metric']!=original_voice['metric'] or
                    voice['original_dnsmos']!=original['original_dnsmos'] or
                    voice['source_sha256']!=old['source_sha256'] or voice['output_sha256']!=item['sha256'] or
                    voice['speaker_minimum']!=quality['speaker_minimum'] or
                    voice['dnsmos_delta']['overall']!=quality['dnsmos_overall_delta']):
                raise ValueError('Voice measurement differs from its operational gate')
            qualification=quality_criteria(old['language'],speech['paired_disagreement'],deltas,eval_deltas,errors,
                voice['speaker_minimum'],voice['dnsmos_delta']['overall'],True,
                old['original_guard_calibration']['language_matches_request'])
            row.update(qualification,detected=observed['detected'],original_reference_error=errors,
                guard_reference_delta=deltas,independent_evaluation=measurements,voice_estimates=voice,
                identical_file_observations_reused=observed['identical_file_observations_reused'],
                qualified_spent_transition=not observed['detected'] and qualification['quality_qualified'])
        rows.append(row)
    successful={r['id'] for r in rows if r['qualified_spent_transition']}
    validated(declaration_path,detector_path)
    if digest(evaluation_path)!=evaluation_sha256 or digest(Path(__file__))!=FORMATTER_SHA256:
        raise ValueError('Evaluation or formatter changed during curation')
    return {'kind':'Spent-source joint TiCodec and middle-band development results',
        'formatter_source_sha256':FORMATTER_SHA256,'declaration_sha256':digest(declaration_path),
        'endpoint_sha256':digest(detector_path),'evaluation_sha256':evaluation_sha256,
        'completed_study_sha256':declaration['completed_study_sha256'],'prior_qualified_pilot':pilot,
        'code_files_sha256':declaration['code_files_sha256'],'evaluation_checkpoints':evaluated['checkpoints'],
        'planned_spent_sources':declaration['planned_sources'],'planned_generation_treatments':len(rows),
        'new_endpoint_treatments':declaration['eligible_treatments'],'qualified_spent_sources':len(successful),
        'treatments':rows,'http_attempts':[{k:v for k,v in c.items() if k!='file'} for c in endpoint['http_attempts']],
        'checks':[{k:v for k,v in c.items() if k!='file'} for c in endpoint['checks']],
        'new_tts_requests':0,'fresh_efficacy_sources':0,'completed_19_of_44_result_changed':False,
        'human_listening_approval':False,'physical_erasure_proven':False,
        'limits':['These are spent development recordings; methods, duplicates and repeats are not independent sources.',
                  'The prior pilot keeps its original measurement date and is not an additional new check.',
                  'Operational voice estimates guide eligibility; Whisper checkpoints share ancestry.',
                  'Estimated quality and negative replies do not prove physical erasure or universal reliability.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('score','export'));parser.add_argument('--declaration',type=Path,required=True)
    parser.add_argument('--detector',type=Path,required=True);parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.phase=='score':score(args.declaration,args.detector,args.evaluation)
    else:
        if args.output is None or args.output.exists():parser.error('Choose a new --output')
        result=export(args.declaration,args.detector,args.evaluation)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('TICODEC_JOINT_QUALIFIED',result['qualified_spent_sources'],'/',result['planned_spent_sources'])
