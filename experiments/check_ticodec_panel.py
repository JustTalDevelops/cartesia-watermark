"""Declare and measure every eligible completed TiCodec/band development file.

All 75 generation treatments remain. The separately qualified exact pilot is
retained as prior evidence; every other eligible unique file gets two new checks.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from codec_audio import digest
from codec_content import assess
from experiments.codec_profile_cache import read_recognition
from experiments.ticodec_generation_report import export as generation_export
from experiments.ticodec_pilot_report import export as pilot_export
from experiments import check_codec_frames as endpoint


def load(path):return json.loads(Path(path).read_text())


def data(band_path,study_path,band_code_root):
    import soundfile as sf
    band_path,study_path,band_code_root=Path(band_path).resolve(),Path(study_path).resolve(),Path(band_code_root).resolve()
    D=study_path.parent.parent;ti_path=D/'ticodec_development/generation.json';frame_path=D/'codec_frame_development/generation.json'
    band,ti,study,frame=load(band_path),load(ti_path),load(study_path),load(frame_path)
    if (band.get('complete') is not True or band.get('error_type') or frame.get('complete') is not True or
            band['completed_study_sha256']!=digest(study_path) or band['ticodec_generation_sha256']!=digest(ti_path) or
            band['frame_generation_sha256']!=digest(frame_path)):
        raise ValueError('Finish and preserve both complete generation panels and native controls')
    verified_ti=generation_export(D)
    pilot=pilot_export(D)
    if not pilot['qualified_spent_transition']:
        raise ValueError('Retain the actual qualified first-source pilot before declaring this follow-up')
    expected=[c for c in study['cases'] if c['marked_provenance'] and not c['qualified_marked_transition']]
    ids=[c['id'] for c in expected]
    if ([c['id'] for c in band['cases']]!=ids or [c['id'] for c in frame['cases']]!=ids or
            band['planned_sources']!=len(ids) or band['planned_treatments']!=2*len(ids)):
        raise ValueError('Keep every failed source and both fixed band mixtures')
    code=dict(ti['code_files_sha256'])
    code.update({name:digest(ROOT/name) for name in ('experiments/check_codec_frames.py',
        'experiments/check_ticodec_panel.py','docs/codec-ticodec-endpoint.md',
        'experiments/ticodec_generation_report.py','experiments/ticodec_pilot_report.py')})
    assessment={}
    def retain(path,expected=None):
        path=Path(path).resolve();checksum=digest(path)
        if expected is not None and checksum!=expected:raise ValueError('A recorded input or assessment changed')
        assessment[str(path)]=checksum
    for name,value in band['code_files_sha256'].items():retain(band_code_root/name,value)
    for p in (ti_path,frame_path,study_path.parent/'original_guards.json',study_path.parent/'original_voice.json',
              D/'ticodec_pilot.declaration.json',D/'ticodec_pilot.assessment.json',D/'ticodec_pilot/result.json',
              D/'ticodec_pilot/quality/evaluation.json',D/'ticodec_pilot/quality/evaluation.plan.json'):
        retain(p)
    voice=load(study_path.parent/'original_voice.json');identity=band['recognition_configuration_sha256']
    if identity!=ti['recognition_configuration_sha256']:raise ValueError('Stem panels used different recognizers')
    sources=[];treatments=[]
    for old,tc,bc,fc in zip(expected,ti['cases'],band['cases'],frame['cases']):
        source=(ROOT/tc['source']).resolve();info=sf.info(source)
        if (digest(source)!=old['source_sha256'] or bc['source_sha256']!=old['source_sha256'] or
                bc['native_baseline_status']!='exact_match' or
                bc['expected_native_sha256']!=fc['treatments'][0]['model']['candidate_flac_sha256']):
            raise ValueError('Source or actual Native control identity changed')
        retain(bc['native_baseline']['file'],bc['expected_native_sha256'])
        original_cache=study_path.parent/'recognition/originals'/(old['source_sha256']+'.json')
        retain(original_cache,old['original_guard_calibration']['recognition_cache_sha256'])
        before=read_recognition(study_path.parent/'recognition/originals',old['source_sha256'],identity)
        sources.append({'id':old['id'],'file':str(source),'sha256':old['source_sha256'],'format':old['source_format']})
        raw=[('ticodec-1g4r',tc,D/'ticodec_development')]+[(t['method'],t,band_path.parent) for t in bc['treatments']]
        if [t['method'] for t in bc['treatments']]!=['ticodec-band-wide','ticodec-band-narrow']:
            raise ValueError('The fixed mixture order changed')
        for method,item,panel in raw:
            output=Path(item['output_file']).resolve();checksum=item['output_sha256'];actual=sf.info(output)
            if (digest(output)!=checksum or not output.is_relative_to(panel.resolve()) or
                    (actual.frames,actual.samplerate,actual.channels)!=(info.frames,info.samplerate,info.channels) or
                    not 5<=info.duration<=300 or max(source.stat().st_size,output.stat().st_size)>32_000_000):
                raise ValueError('Use matching unchanged whole files within service limits')
            cached=panel/'recognition/processed'/(checksum+'.json');retain(cached,item['recognition_cache_sha256'])
            after=read_recognition(panel/'recognition/processed',checksum,identity)
            speech=assess(before,after)
            if speech!=item['speech_assessment']:raise ValueError('Speech evidence differs from its actual cache')
            status=item['status'];quality=item.get('runtime_quality_guard')
            if status=='content_rejected':
                if speech['content_guard_passed'] or quality is not None:raise ValueError('Inconsistent speech rejection')
            elif status in ('processing_eligible','runtime_quality_rejected'):
                receipt_path=panel/'runtime_voice'/old['id']/(checksum+'.json');retain(receipt_path,quality['receipt_sha256'])
                receipt=load(receipt_path)
                if len(receipt['clips'])!=1:raise ValueError('Keep exactly one voice observation per file')
                clip=receipt['clips'][0]
                original=next(c for c in voice['clips'] if c['id']==old['id'])
                passing=clip['speaker_minimum']>=.95 and clip['dnsmos_delta']['overall']>=-.2
                if (not speech['content_guard_passed'] or receipt.get('complete') is not True or
                        receipt['metric']!=voice['metric'] or clip['original_dnsmos']!=original['original_dnsmos'] or
                        clip['source_sha256']!=old['source_sha256'] or clip['output_sha256']!=checksum or
                        clip['status']!='scored' or quality['source_sha256']!=old['source_sha256'] or
                        quality['candidate_sha256']!=checksum or type(quality['passed']) is not bool or
                        quality['passed']!=passing or passing!=(status=='processing_eligible') or
                        quality['speaker_minimum']!=clip['speaker_minimum'] or
                        quality['dnsmos_overall_delta']!=clip['dnsmos_delta']['overall']):
                    raise ValueError('Voice evidence or frozen eligibility limits changed')
            else:raise ValueError('An input panel contains unfinished work')
            reused=(status=='processing_eligible' and checksum==pilot['candidate_sha256'] and
                    old['source_sha256']==pilot['source_sha256'] and old['id']==pilot['case_id'])
            row={'id':old['id'],'method':method,'sha256':checksum,'file':str(output),'processing_status':status,
                 'detector_eligible':status=='processing_eligible' and not reused,
                 'endpoint_phase_status':'qualified_pilot_reused' if reused else 'pending' if status=='processing_eligible' else 'processing_rejected'}
            if reused:row.update(prior_pilot_endpoint_sha256=pilot['endpoint_sha256'],prior_quality_qualified=True,
                                 prior_candidate_detected=False,new_requests_for_this_treatment=0)
            treatments.append(row)
    known=endpoint.previous_observations(study)
    for check in load(D/'ticodec_pilot/result.json')['checks']:
        checksum=check['sha256'];detected=check['detected']
        if checksum in known and known[checksum]!=detected:raise ValueError('Historical complete-phase receipts conflict')
        known[checksum]=detected
    pending=[t for t in treatments if t['detector_eligible']];unique=len({t['sha256'] for t in pending})
    return {'kind':'Declared spent-source TiCodec and band endpoint panel',
        'observation_report_kind':'Spent-source TiCodec and band detector observations','progress_prefix':'TICODEC_PANEL_DETECTOR',
        'generation_file':str(band_path),'generation_sha256':digest(band_path),'band_code_root':str(band_code_root),
        'ticodec_generation_file':str(ti_path),'ticodec_generation_sha256':digest(ti_path),
        'completed_study_file':str(study_path),'completed_study_sha256':digest(study_path),
        'code_files_sha256':code,'assessment_files_sha256':assessment,'sources':sources,'treatments':treatments,
        'historical_exact_file_observations':known,'minimum_detector_start_interval_seconds':80,
        'planned_sources':len(sources),'planned_generation_treatments':len(treatments),
        'operationally_eligible_treatments':sum(t['processing_status']=='processing_eligible' for t in treatments),
        'qualified_pilot_treatments_reused':sum(t['endpoint_phase_status']=='qualified_pilot_reused' for t in treatments),
        'eligible_treatments':len(pending),'unique_eligible_files':unique,'maximum_http_attempts':2*unique+4,
        'repetitions_per_exact_file':2,'opening_source_id':sources[0]['id'],'closing_source_id':sources[-1]['id'],
        'prior_pilot_independent_quality_qualified':True,'new_tts_requests':0,'fresh_efficacy_sources':0,
        'reference_evaluation_order':'After the complete newly declared endpoint phase; score every newly eligible treatment including positives',
        'human_listening_approval':False,'physical_erasure_proven':False}


def declare(band_path,study_path,band_code_root,path):
    if path.exists():raise ValueError('Never overwrite a declared endpoint inventory')
    value=data(band_path,study_path,band_code_root);value['declared_at']=endpoint.now()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return value


def run(declaration_path,output_path):
    declaration=load(declaration_path)
    builder=lambda band,study:data(band,study,Path(declaration['band_code_root']))
    return endpoint.run(declaration_path,output_path,builder=builder)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='phase',required=True)
    before=sub.add_parser('declare');before.add_argument('--band-generation',type=Path,required=True)
    before.add_argument('--completed-study',type=Path,required=True)
    before.add_argument('--band-code-root',type=Path,required=True);before.add_argument('--declaration',type=Path,required=True)
    after=sub.add_parser('detect');after.add_argument('--declaration',type=Path,required=True);after.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.phase=='declare':
        result=declare(args.band_generation,args.completed_study,args.band_code_root,args.declaration)
        print('TICODEC_PANEL_DECLARED',result['planned_generation_treatments'],'rows;',result['eligible_treatments'],'new treatments;',result['maximum_http_attempts'],'HTTP maximum')
    else:run(args.declaration,args.output)
