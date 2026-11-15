"""Curate the measured spent TiCodec pilot without running models or API calls.

Requires the original complete calibration, actual six-request endpoint report,
processing caches, voice receipt and separately completed reference evaluation.
"""
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from codec_audio import digest
from codec_content import assess
from codec_text import paired_error,text_units
from experiments.codec_profile_cache import read_recognition
from experiments.codec_profile_evaluation import quality_criteria


def load(path):return json.loads(path.read_text())


def export(directory):
    D=Path(directory)
    pilot=D/'ticodec_pilot';declaration=load(D/'ticodec_pilot.declaration.json')
    endpoint=load(pilot/'result.json');ti=load(D/'ticodec_development/generation.json')
    if (endpoint.get('complete') is not True or endpoint.get('error_type') or
            ti.get('complete') is not True or ti.get('error_type') or
            type(endpoint.get('candidate_detected')) is not bool or
            endpoint['source_sha256']!=declaration['source_sha256'] or
            endpoint['candidate_sha256']!=declaration['candidate_sha256'] or
            endpoint['assessment_sha256']!=declaration['assessment_sha256'] or
            endpoint['checker_source_sha256']!=declaration['helper_source_sha256'] or
            len(endpoint['checks'])!=6 or len(endpoint['http_attempts'])!=6):
        raise ValueError('Complete the declared six-request pilot and TiCodec generation first')
    expected=[('opening_original',1),('opening_original',2),('candidate',1),('candidate',2),
              ('closing_original',1),('closing_original',2)]
    starts=[]
    for (role,ordinal),check,attempt in zip(expected,endpoint['checks'],endpoint['http_attempts']):
        checksum=declaration['candidate_sha256'] if role=='candidate' else declaration['source_sha256']
        if (any(check[k]!=v or attempt[k]!=v for k,v in [('role',role),('ordinal',ordinal),('sha256',checksum)]) or
                check['http_status']!=200 or attempt['http_status']!=200 or type(check.get('detected')) is not bool or
                check['detected']!=(endpoint['candidate_detected'] if role=='candidate' else True)):
            raise ValueError('Pilot receipts conflict or refer to different exact files')
        starts.append(datetime.fromisoformat(attempt['started_at']))
    if (starts[0]<datetime.fromisoformat(declaration['declared_at']) or
            any((b-a).total_seconds()<79.9 for a,b in zip(starts,starts[1:]))):
        raise ValueError('Pilot declaration timing or shared pacing changed')
    source,candidate=Path(declaration['source_file']),Path(declaration['candidate_file'])
    if digest(source)!=declaration['source_sha256'] or digest(candidate)!=declaration['candidate_sha256']:
        raise ValueError('Pilot input bytes changed')
    import soundfile as sf
    first,second=sf.info(source),sf.info(candidate)
    if (first.frames,first.samplerate,first.channels)!=(second.frames,second.samplerate,second.channels):
        raise ValueError('Pilot output changed the complete original sample grid')
    case=next(c for c in ti['cases'] if c['id']==declaration['case_id'])
    if case['status']!='processing_eligible' or case['output_sha256']!=digest(candidate):
        raise ValueError('Pilot no longer binds its operationally assessed candidate')
    F=D/'codec_profiles_fresh';study=load(F/'curated_complete_report.json')
    old=next(c for c in study['cases'] if c['id']==case['id'])
    if (digest(F/'curated_complete_report.json')!=ti['completed_study_sha256'] or
            case['id']!=ti['cases'][0]['id'] or old['marked_provenance'] is not True or
            old['original_detected'] is not True or old['qualified_marked_transition'] is not False):
        raise ValueError('Pilot must remain the first spent marked failure in its completed study')
    authored=next(i for i in load(F/'evaluation.plan.json')['items'] if i['id']==case['id'])
    folder=pilot/'quality';plan=folder/'evaluation.plan.json';evaluation=folder/'evaluation.json'
    expected_plan={'items':[{'id':case['id'],'source':str(source),'sha256':digest(source),
        'output':str(candidate),'language':case['language'],'text':authored['text']}]}
    if load(plan)!=expected_plan or load(evaluation)['plan_sha256']!=digest(plan):
        raise ValueError('Evaluation plan no longer binds the authored reference and exact files')
    measured=load(evaluation);oe=load(F/'original_evaluation.json')
    original=next(c for c in oe['clips'] if c['id']==case['id'] and c['metric']=='small')
    if (not measured['complete'] or len(measured['clips'])!=1 or
            measured['checkpoints']!={'small':oe['checkpoints']['small']}):
        raise ValueError('The pilot evaluator differs from the frozen Small checkpoint')
    item=measured['clips'][0]
    if (item['status']!='scored' or item['original']!=original['original'] or
            item['source_sha256']!=digest(source) or item['output_sha256']!=digest(candidate)):
        raise ValueError('Pilot reference evaluation refers to different whole files')
    identity=ti['recognition_configuration_sha256']
    processed_cache=D/'ticodec_development/recognition/processed'/(digest(candidate)+'.json')
    if digest(processed_cache)!=case['recognition_cache_sha256']:
        raise ValueError('Candidate recognition changed after generation')
    original_cache=F/'recognition/originals'/(digest(source)+'.json')
    if digest(original_cache)!=old['original_guard_calibration']['recognition_cache_sha256']:
        raise ValueError('Original guard calibration changed')
    before=read_recognition(F/'recognition/originals',digest(source),identity)
    after=read_recognition(D/'ticodec_development/recognition/processed',digest(candidate),identity)
    speech=assess(before,after)
    if speech!=case['speech_assessment'] or not speech['content_guard_passed']:
        raise ValueError('Actual processing caches disagree with the pilot assessment')
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    truth=text_units(authored['text'],BasicTextNormalizer(remove_diacritics=False),case['language'])
    guards=(speech['primary_guard'],speech['secondary_guard'])
    errors={name:paired_error(truth,before[name],case['language']) for name in guards}
    if errors!=old['original_guard_calibration']['original_reference_error']:
        raise ValueError('Original guard reference errors changed')
    deltas={}
    for name in guards:
        actual=paired_error(truth,after[name],case['language'])
        deltas[name]=None if actual is None or errors[name] is None else actual-errors[name]
    errors['small']=original['original']['reference_error']
    actual=item['processed']['reference_error']
    delta=None if actual is None or errors['small'] is None else actual-errors['small']
    if delta!=item['reference_error_delta']:raise ValueError('Independent reference delta changed')
    voice_path=D/'ticodec_development/runtime_voice'/case['id']/(digest(candidate)+'.json')
    receipt=load(voice_path);quality=case['runtime_quality_guard'];ov=load(F/'original_voice.json')
    voice=receipt['clips'][0];original_voice=next(c for c in ov['clips'] if c['id']==case['id'])
    if (digest(voice_path)!=quality['receipt_sha256'] or receipt['metric']!=ov['metric'] or not receipt['complete'] or
            voice['source_sha256']!=digest(source) or voice['output_sha256']!=digest(candidate) or
            voice['original_dnsmos']!=original_voice['original_dnsmos'] or
            voice['speaker_minimum']!=quality['speaker_minimum'] or
            voice['dnsmos_delta']['overall']!=quality['dnsmos_overall_delta']):
        raise ValueError('Pilot voice evidence differs from the actual operational estimates')
    qualification=quality_criteria(case['language'],speech['paired_disagreement'],deltas,{'small':delta},errors,
        voice['speaker_minimum'],voice['dnsmos_delta']['overall'],True,
        old['original_guard_calibration']['language_matches_request'])
    if digest(source)!=declaration['source_sha256'] or digest(candidate)!=declaration['candidate_sha256']:
        raise ValueError('Pilot whole files changed during scoring')
    result={'kind':'Single spent-source TiCodec pilot','case_id':case['id'],
        'source_sha256':digest(source),'candidate_sha256':digest(candidate),
        'completed_study_sha256':digest(F/'curated_complete_report.json'),
        'ticodec_generation_sha256':digest(D/'ticodec_development/generation.json'),
        'declaration_sha256':digest(D/'ticodec_pilot.declaration.json'),'endpoint_sha256':digest(pilot/'result.json'),
        'evaluation_sha256':digest(evaluation),'formatter_sha256':digest(Path(__file__)),
        'source_format':old['source_format'],'model':ti['codec'],'speech_assessment':speech,
        'original_reference_error':errors,'guard_reference_delta':deltas,'independent_evaluation':[item],
        'voice_estimates':voice,**qualification,'candidate_detected':endpoint['candidate_detected'],
        'qualified_spent_transition':not endpoint['candidate_detected'] and qualification['quality_qualified'],
        'http_attempts':endpoint['http_attempts'],'checks':endpoint['checks'],
        'new_tts_requests':0,'fresh_efficacy_sources':0,'completed_19_of_44_result_changed':False,
        'human_listening_approval':False,'physical_erasure_proven':False,
        'limits':['One spent source; repeated calls do not add independent sources.',
                  'Operational voice estimates guide eligibility; Whisper checkpoints share ancestry.',
                  'Negative detector replies and estimated quality do not prove physical erasure or universal removal.']}
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new report; existing observations are not overwritten')
    result=export(args.directory)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('TICODEC_PILOT_EXPORTED',result['qualified_spent_transition'])
