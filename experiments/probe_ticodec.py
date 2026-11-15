"""Generate one fixed TiCodec treatment for every spent marked study failure.

No detector/TTS requests or reference-text-guided generation. All rendered files,
processing rejections and operational voice rejections remain in the manifest.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from codec_audio import digest,quantize
from codec_content import assess
from experiments.ticodec_adapter import TiCodec,ADAPTER_SHA256,ASSETS_SHA256
from experiments.probe_codec_frames import completed_study,failed_marked_cases,without_http,save


def run(study_path,parity_path,directory):
    if directory.exists() or directory.resolve().is_relative_to(study_path.parent.resolve()):
        raise ValueError('Choose a new development directory outside the completed study')
    parity=json.loads(parity_path.read_text())
    if (parity.get('complete') is not True or parity.get('error_type') or
            parity.get('exact_author_api_waveform') is not True or parity['adapter_sha256']!=ADAPTER_SHA256 or
            parity['assets_manifest_sha256']!=ASSETS_SHA256 or
            parity['validation_code_sha256']!=digest(ROOT/'experiments/validate_ticodec.py')):
        raise ValueError('Complete matching synthetic author-API parity before generating speech')
    with without_http():
        import soundfile as sf
        from experiments.codec_profile_cache import StudyRecognizer
        from experiments.run_codec_profiles import runtime_voice_decision
        study,plan,freeze_path=completed_study(study_path)
        freeze=json.loads(freeze_path.read_text())
        old=json.loads((study_path.parent/'original_guards.json').read_text())
        voice=json.loads((study_path.parent/'original_voice.json').read_text())
        originals={c['source_sha256']:c['recognition_cache_sha256'] for c in old['clips']}
        inputs={i['id']:i for i in plan['items']}
        registry={name:digest(ROOT/name) for name in list(freeze['source_files_sha256'])+[
            'experiments/ticodec_adapter.py','experiments/ticodec_assets.json','experiments/validate_ticodec.py',
            'experiments/probe_ticodec.py','experiments/probe_codec_frames.py','experiments/codec_profiles_report.py']}
        cases=[{'id':c['id'],'source':inputs[c['id']]['source'],'source_sha256':c['source_sha256'],
            'language':c['language'],'detected_language':c['detected_language'],'status':'planned'}
            for c in failed_marked_cases(study)]
        report={'kind':'Spent-source TiCodec 1g4r generation panel','complete':False,
            'completed_study_sha256':digest(study_path),'parity_report_sha256':digest(parity_path),
            'freeze_sha256':digest(freeze_path),'code_files_sha256':registry,
            'planned_spent_sources':len(cases),'planned_treatments':len(cases),'cases':cases,
            'new_tts_requests':0,'new_http_requests':0,'fresh_efficacy_sources':0,
            'reference_text_used_for_generation_or_repair':False,'detector_checked':False,
            'independent_quality_evaluated':False,'human_listening_approval':False,
            'physical_erasure_proven':False,'started_at':datetime.now(timezone.utc).isoformat()}
        directory.mkdir(parents=True)
        manifest=directory/'generation.json'
        save(manifest,report)
        try:
            started=time.perf_counter();codec=TiCodec()
            report.update(codec=codec.metadata,codec_load_seconds=time.perf_counter()-started)
            recognizer=StudyRecognizer(study_path.parent/'recognition/originals',directory/'recognition/processed',
                source_digests=originals)
            if recognizer.identity!=old['configuration_sha256']:
                raise ValueError('Processing recognizers differ from original-only calibration')
            report.update(recognizers=recognizer.metadata,recognition_configuration_sha256=recognizer.identity)
            save(manifest,report)
            for case in cases:
                if any(digest(ROOT/name)!=value for name,value in registry.items()):
                    raise ValueError('Declared generator or frozen processing code changed')
                source=ROOT/case['source']
                if digest(source)!=case['source_sha256']:raise ValueError('Spent source changed')
                case['status']='processing';save(manifest,report)
                audio,rate=sf.read(source,dtype='float32')
                before=recognizer.transcribe(source)
                if before['language']!=case['detected_language']:
                    raise ValueError('Original-only detected language changed')
                started=time.perf_counter();rebuilt,gain=codec.reconstruct(audio,rate)
                folder=directory/case['id'];folder.mkdir()
                output=folder/'ticodec-1g4r.flac'
                # Every render is retained privately, including failed speech.
                quantize(output,rebuilt,rate)
                case.update(output_file=str(output),output_sha256=digest(output),gain=gain,
                    codec_elapsed_seconds=time.perf_counter()-started,
                    source_format={'samples':len(audio),'sample_rate':rate,'channels':1 if audio.ndim==1 else audio.shape[1]})
                started=time.perf_counter();after=recognizer.transcribe_candidate(output,before)
                if digest(output)!=case['output_sha256']:raise ValueError('Assessed encoded file changed')
                speech=assess(before,after)
                case.update(speech_assessment=speech,recognition_seconds=time.perf_counter()-started,
                    recognition_cache_sha256=digest(directory/'recognition/processed'/(case['output_sha256']+'.json')))
                if not speech['content_guard_passed']:
                    case['status']='content_rejected'
                else:
                    quality=runtime_voice_decision(directory,case['id'],case['source_sha256'],
                        case['detected_language'],voice,source,output)
                    case.update(runtime_quality_guard=quality,
                        status='processing_eligible' if quality['passed'] else 'runtime_quality_rejected')
                if (digest(source)!=case['source_sha256'] or digest(study_path)!=report['completed_study_sha256'] or
                        any(digest(ROOT/name)!=value for name,value in registry.items())):
                    raise ValueError('Original evidence or generation code changed')
                save(manifest,report)
                print('TICODEC_GENERATED',case['id'],case['status'],flush=True)
            report.update(complete=True,completed_at=datetime.now(timezone.utc).isoformat())
            save(manifest,report)
        except BaseException as error:
            report.update(error_type=type(error).__name__,stopped_at=datetime.now(timezone.utc).isoformat())
            save(manifest,report);raise
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-study',type=Path,required=True)
    parser.add_argument('--parity',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();run(args.completed_study,args.parity,args.directory)
