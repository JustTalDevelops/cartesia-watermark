"""Export every TiCodec generation outcome without running models or detection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from codec_audio import digest
from codec_content import assess
from experiments.codec_profile_cache import read_recognition


def load(path):return json.loads(path.read_text())


def export(directory):
    import soundfile as sf
    directory=Path(directory);panel=directory/'ticodec_development';F=directory/'codec_profiles_fresh'
    measured=load(panel/'generation.json');study=load(F/'curated_complete_report.json')
    expected=[c for c in study['cases'] if c['marked_provenance'] and not c['qualified_marked_transition']]
    if (measured.get('complete') is not True or measured.get('error_type') or
            measured['completed_study_sha256']!=digest(F/'curated_complete_report.json') or
            [c['id'] for c in measured['cases']]!=[c['id'] for c in expected] or
            measured['planned_spent_sources']!=len(expected) or measured['planned_treatments']!=len(expected)):
        raise ValueError('Keep the complete generation panel and every spent marked failure')
    if any(digest(ROOT/name)!=value for name,value in measured['code_files_sha256'].items()):
        raise ValueError('Recorded generation or frozen processing sources changed')
    identity=measured['recognition_configuration_sha256'];voice=load(F/'original_voice.json');rows=[]
    for case,old in zip(measured['cases'],expected):
        source=ROOT/case['source'];output=Path(case['output_file'])
        first,second=sf.info(source),sf.info(output)
        grid={'samples':first.frames,'sample_rate':first.samplerate,'channels':first.channels}
        if (digest(source)!=case['source_sha256'] or case['source_sha256']!=old['source_sha256'] or
                digest(output)!=case['output_sha256'] or not output.resolve().is_relative_to(panel.resolve()) or
                grid!=case['source_format'] or grid!=old['source_format'] or
                (first.frames,first.samplerate,first.channels)!=(second.frames,second.samplerate,second.channels)):
            raise ValueError('Generated evidence no longer binds matching complete files')
        original_cache=F/'recognition/originals'/(case['source_sha256']+'.json')
        processed_cache=panel/'recognition/processed'/(case['output_sha256']+'.json')
        if (digest(original_cache)!=old['original_guard_calibration']['recognition_cache_sha256'] or
                digest(processed_cache)!=case['recognition_cache_sha256']):
            raise ValueError('Source or candidate recognition changed after measurement')
        before=read_recognition(F/'recognition/originals',case['source_sha256'],identity)
        after=read_recognition(panel/'recognition/processed',case['output_sha256'],identity)
        speech=assess(before,after)
        if speech!=case['speech_assessment']:raise ValueError('Speech assessment differs from its actual cache')
        status=case['status'];quality=case.get('runtime_quality_guard')
        if status=='content_rejected':
            if speech['content_guard_passed'] or quality is not None:
                raise ValueError('A speech rejection has inconsistent operational evidence')
        elif status in ('processing_eligible','runtime_quality_rejected'):
            receipt_path=panel/'runtime_voice'/case['id']/(case['output_sha256']+'.json')
            receipt=load(receipt_path);clip=receipt['clips'][0]
            original=next(c for c in voice['clips'] if c['id']==case['id'])
            if (not speech['content_guard_passed'] or digest(receipt_path)!=quality['receipt_sha256'] or
                    receipt.get('complete') is not True or receipt['metric']!=voice['metric'] or
                    clip['original_dnsmos']!=original['original_dnsmos'] or
                    clip['source_sha256']!=case['source_sha256'] or clip['output_sha256']!=case['output_sha256'] or
                    clip['speaker_minimum']!=quality['speaker_minimum'] or
                    clip['dnsmos_delta']['overall']!=quality['dnsmos_overall_delta']):
                raise ValueError('Voice gate differs from the actual operational measurement')
            passing=clip['speaker_minimum']>=.95 and clip['dnsmos_delta']['overall']>=-.2
            if type(quality['passed']) is not bool or quality['passed']!=passing or passing!=(status=='processing_eligible'):
                raise ValueError('Operational status differs from the unchanged voice limits')
        else:raise ValueError('A completed panel contains unfinished or unknown work')
        row={k:case[k] for k in ('id','language','detected_language','source_sha256','output_sha256',
            'source_format','status','gain','codec_elapsed_seconds','recognition_seconds','speech_assessment',
            'recognition_cache_sha256')}
        row.update(source_model=old['model'],voice_id=old['voice'],
            source_recognition_cache_sha256=digest(original_cache),runtime_quality_guard=quality)
        rows.append(row)
    return {'kind':'Spent-source TiCodec 1g4r generation results only',
        'generation_sha256':digest(panel/'generation.json'),'formatter_source_sha256':digest(Path(__file__)),
        'completed_study_sha256':measured['completed_study_sha256'],'model':measured['codec'],
        'code_files_sha256':measured['code_files_sha256'],'planned_spent_sources':len(rows),
        'planned_treatments':len(rows),'operational_status_counts':dict(Counter(c['status'] for c in rows)),
        'recognition_configuration_sha256':identity,'cases':rows,
        'generator_new_http_requests':0,'new_tts_requests':0,'fresh_efficacy_sources':0,
        'independent_quality_evaluated_by_generation':False,'completed_study_result_changed':False,
        'human_listening_approval':False,'physical_erasure_proven':False,
        'limits':['All recordings are spent failures of the completed 19/44 study.',
                  'Processing eligibility is not removal or independent quality qualification.',
                  'Endpoint and reference measurements of a separate pilot are reported separately.',
                  'Operational voice estimates guide eligibility and are not independent confirmation.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new report')
    report=export(args.directory)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('TICODEC_GENERATION_EXPORTED',report['operational_status_counts'])
