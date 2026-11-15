"""Generate the fixed 3x25 spent-source TiCodec latent ablation with exact controls."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from codec_audio import digest,quantize
from experiments.ticodec_latents import TiCodecLatents,MODES,IMPLEMENTATION_SHA256
from experiments.probe_codec_frames import completed_study,failed_marked_cases,without_http,save

DRIVER_SHA256=digest(Path(__file__))


def planned_cases(study,previous):
    failed=failed_marked_cases(study)
    if [c['id'] for c in failed]!=[c['id'] for c in previous['cases']]:
        raise ValueError('Retain every spent marked failure in its original order')
    return [{'id':old['id'],'source':old['source'],'source_sha256':case['source_sha256'],
        'language':case['language'],'detected_language':case['detected_language'],
        'control_expected_sha256':old['output_sha256'],'control_status':'planned',
        'treatments':[{'mode':name,'local_quantized':local,'global_quantized':global_,'status':'planned'}
            for name,local,global_ in MODES if name!='quantized-control']}
        for case,old in zip(failed,previous['cases'])]


def run(study_path,previous_path,directory,model_factory=None):
    if directory.exists() or any(directory.resolve().is_relative_to(p.parent.resolve()) for p in (study_path,previous_path)):
        raise ValueError('Choose a new latent-panel directory outside completed input panels')
    previous=json.loads(previous_path.read_text())
    if previous.get('complete') is not True or previous.get('error_type'):
        raise ValueError('Finish the original TiCodec panel before the latent ablation')
    with without_http():
        import numpy as np
        import soundfile as sf
        from codec_content import assess
        from experiments.codec_profile_cache import StudyRecognizer
        from experiments.run_codec_profiles import runtime_voice_decision
        study,plan,freeze_path=completed_study(study_path)
        if digest(study_path)!=previous['completed_study_sha256']:
            raise ValueError('The TiCodec panel belongs to another completed study')
        if any(digest(ROOT/name)!=value for name,value in previous['code_files_sha256'].items()):
            raise ValueError('Original TiCodec or frozen processing source changed')
        cases=planned_cases(study,previous)
        for case,old in zip(cases,previous['cases']):
            if (case['source_sha256']!=old['source_sha256'] or digest(ROOT/case['source'])!=case['source_sha256'] or
                    digest(Path(old['output_file']))!=case['control_expected_sha256']):
                raise ValueError('Use the unchanged complete source and original TiCodec control')
        original=json.loads((study_path.parent/'original_guards.json').read_text())
        original_voice=json.loads((study_path.parent/'original_voice.json').read_text())
        registry=dict(previous['code_files_sha256'])
        registry.update({name:digest(ROOT/name) for name in ('experiments/ticodec_latents.py',
            'experiments/probe_ticodec_latents.py','docs/codec-ticodec-latents.md')})
        if (registry['experiments/ticodec_latents.py']!=IMPLEMENTATION_SHA256 or
                registry['experiments/probe_ticodec_latents.py']!=DRIVER_SHA256):
            raise ValueError('Loaded latent helper or driver changed')
        report={'kind':'Spent-source TiCodec local/global quantization ablation','complete':False,
            'completed_study_sha256':digest(study_path),'original_ticodec_generation_sha256':digest(previous_path),
            'code_files_sha256':registry,'planned_spent_sources':len(cases),'planned_treatments':3*len(cases),
            'planned_exact_file_controls':len(cases),'cases':cases,'new_tts_requests':0,'new_http_requests':0,
            'fresh_efficacy_sources':0,'reference_text_used_for_generation_or_repair':False,
            'independent_quality_evaluated':False,'detector_checked':False,'human_listening_approval':False,
            'physical_erasure_proven':False,'started_at':datetime.now(timezone.utc).isoformat()}
        directory.mkdir(parents=True);manifest=directory/'generation.json';save(manifest,report)
        def unchanged():
            if (digest(previous_path)!=report['original_ticodec_generation_sha256'] or
                    digest(study_path)!=report['completed_study_sha256'] or
                    any(digest(ROOT/name)!=value for name,value in registry.items())):
                raise ValueError('Latent generator or completed input evidence changed')
        try:
            codec=(model_factory or TiCodecLatents)()
            report['codec']=codec.metadata
            # Real weights, synthetic signal, original adapter API as control.
            native=np.asarray(.12*np.sin(2*np.pi*317*np.arange(120000,dtype=np.float32)/24000),dtype=np.float32)
            prepared=codec.prepare(native)
            control=codec.decode(prepared,'quantized-control');reference=codec.base.decode(native)
            exact=bool(np.array_equal(control,reference))
            report['synthetic_control']={'exact_original_adapter_waveform':exact,
                'maximum_absolute_error':float(np.max(np.abs(control-reference))),
                'speech_quality_evaluated':False,'detector_checked':False}
            save(manifest,report)
            if not exact:raise ValueError('Synthetic quantized control differs from the original adapter')
            recognizer=StudyRecognizer(study_path.parent/'recognition/originals',directory/'recognition/processed',
                source_digests={c['source_sha256']:c['recognition_cache_sha256'] for c in original['clips']})
            if recognizer.identity!=original['configuration_sha256']:
                raise ValueError('Recognizer configuration differs from original source calibration')
            report.update(recognizers=recognizer.metadata,recognition_configuration_sha256=recognizer.identity)
            for case in cases:
                unchanged();source=ROOT/case['source'];folder=directory/case['id'];folder.mkdir()
                case['control_status']='processing';save(manifest,report)
                audio,rate=sf.read(source,dtype='float32')
                def check_control(wave,gain):
                    target=folder/'quantized-control.flac';quantize(target,wave,rate)
                    checksum=digest(target)
                    case.update(control_actual_sha256=checksum,control_file=str(target),control_gain=gain,
                        control_status='exact_match' if checksum==case['control_expected_sha256'] else 'mismatch')
                    save(manifest,report)
                    if checksum!=case['control_expected_sha256']:
                        raise ValueError('Actual quantized control differs from its original whole-file bytes')
                rendered=codec.reconstruct_modes(audio,rate,check_control)
                before=recognizer.transcribe(source)
                for treatment in case['treatments']:
                    treatment['status']='processing';save(manifest,report)
                    output=folder/(treatment['mode']+'.flac');wave,gain=rendered[treatment['mode']]
                    quantize(output,wave,rate);checksum=digest(output)
                    treatment.update(output_file=str(output),output_sha256=checksum,gain=gain)
                    after=recognizer.transcribe_candidate(output,before)
                    speech=assess(before,after)
                    treatment.update(speech_assessment=speech,
                        recognition_cache_sha256=digest(directory/'recognition/processed'/(checksum+'.json')))
                    if not speech['content_guard_passed']:treatment['status']='content_rejected'
                    else:
                        quality=runtime_voice_decision(directory,case['id'],case['source_sha256'],
                            case['detected_language'],original_voice,source,output)
                        treatment.update(runtime_quality_guard=quality,
                            status='processing_eligible' if quality['passed'] else 'runtime_quality_rejected')
                    if digest(source)!=case['source_sha256'] or digest(output)!=checksum:
                        raise ValueError('Source or assessed exact candidate changed')
                    unchanged();save(manifest,report)
                    print('TICODEC_LATENT_GENERATED',case['id'],treatment['mode'],treatment['status'],flush=True)
            report.update(complete=True,completed_at=datetime.now(timezone.utc).isoformat());save(manifest,report)
        except BaseException as error:
            report.update(error_type=type(error).__name__,stopped_at=datetime.now(timezone.utc).isoformat())
            save(manifest,report);raise
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-study',type=Path,required=True)
    parser.add_argument('--original-ticodec-generation',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();run(args.completed_study,args.original_ticodec_generation,args.directory)
