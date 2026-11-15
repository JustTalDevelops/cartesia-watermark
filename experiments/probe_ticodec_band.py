"""Assess two fixed Native/TiCodec band mixtures on all 25 spent failures.

No reference-text-guided generation or detector/TTS calls. Every rendered file
and rejection is retained. Native stems must reproduce the completed frame
panel's zero-offset bytes before a mixture is generated for that source.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from codec_audio import digest,mix_reconstructions,quantize

IMPLEMENTATION_SHA256=digest(Path(__file__))
BANDS=(('ticodec-band-wide',1000,4000),('ticodec-band-narrow',1500,3500))

class NativeBaselineMismatch(ValueError):
    pass


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


@contextmanager
def without_http():
    import requests
    original=requests.sessions.Session.request
    def blocked(*args,**kwargs):raise RuntimeError('HTTP is disabled for band generation')
    requests.sessions.Session.request=blocked
    previous={name:os.environ.get(name) for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    try:yield
    finally:
        requests.sessions.Session.request=original
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def cases_for(study,ti,frames):
    failed=[c for c in study['cases'] if c['marked_provenance'] and not c['qualified_marked_transition']]
    ids=[c['id'] for c in failed]
    if not ids or [c['id'] for c in ti['cases']]!=ids or [c['id'] for c in frames['cases']]!=ids:
        raise ValueError('Keep every spent marked failure in its original order')
    rows=[]
    for original,donor,frame in zip(failed,ti['cases'],frames['cases']):
        zero=frame['treatments'][0]
        if (original['source_sha256']!=donor['source_sha256'] or donor['source_sha256']!=frame['source_sha256'] or
                zero['offset_native_samples']!=0):
            raise ValueError('Stems refer to different sources or a nonzero native baseline')
        rows.append({'id':original['id'],'language':original['language'],'detected_language':original['detected_language'],
            'source':donor['source'],'source_sha256':original['source_sha256'],
            'source_format':original['source_format'],'ticodec_file':donor['output_file'],
            'ticodec_sha256':donor['output_sha256'],'ticodec_original_processing_status':donor['status'],
            'expected_native_sha256':zero['model']['candidate_flac_sha256'],'native_baseline_status':'planned',
            'treatments':[{'method':name,'band_hz':[low,high],'status':'planned'} for name,low,high in BANDS]})
    return rows


def native_stem(codec,source,output,expected):
    import soundfile as sf
    audio,rate=sf.read(source,dtype='float32')
    started=time.perf_counter();rebuilt,gain=codec.reconstruct(audio,rate,bandwidth_kbps=12.)
    canonical=quantize(output,rebuilt,rate)
    if digest(output)!=expected:
        raise NativeBaselineMismatch('Regenerated Native stem differs from the exact zero-offset frame baseline')
    return canonical,rate,{'sha256':expected,'gain':gain,'elapsed_seconds':time.perf_counter()-started,
                          'baseline_comparison':'exact_match','file':str(output)}


def run(study_path,ti_path,frame_path,directory,codec_factory=None):
    if directory.exists() or any(directory.resolve().is_relative_to(p.parent.resolve()) for p in (study_path,ti_path,frame_path)):
        raise ValueError('Choose a new band directory outside all completed panels')
    ti,frames=json.loads(ti_path.read_text()),json.loads(frame_path.read_text())
    if any(v.get('complete') is not True or v.get('error_type') for v in (ti,frames)):
        raise ValueError('Finish both stem-generation panels before mixing')
    with without_http():
        import soundfile as sf
        from codec_content import assess
        from codec_inference import NativeCodec
        from experiments.codec_profile_cache import StudyRecognizer
        from experiments.codec_profiles_report import export
        from experiments.run_codec_profiles import runtime_voice_decision
        study=json.loads(study_path.read_text())
        freeze_path=ROOT/'results/codec_profiles_reliability.freeze.json'
        freeze=json.loads(freeze_path.read_text())
        if (digest(study_path)!=ti['completed_study_sha256'] or digest(study_path)!=frames['completed_study_sha256'] or
                export(freeze_path,study_path.parent)!=study):
            raise ValueError('Use the unchanged completed study and its actual stem panels')
        cases=cases_for(study,ti,frames)
        for case in cases:
            source=ROOT/case['source'];donor=Path(case['ticodec_file'])
            info,actual=sf.info(source),sf.info(donor)
            grid={'samples':info.frames,'sample_rate':info.samplerate,'channels':info.channels}
            if (digest(source)!=case['source_sha256'] or digest(donor)!=case['ticodec_sha256'] or
                    not donor.resolve().is_relative_to(ti_path.parent.resolve()) or grid!=case['source_format'] or
                    (actual.frames,actual.samplerate,actual.channels)!=(info.frames,info.samplerate,info.channels)):
                raise ValueError('Use complete unchanged source and TiCodec stem files')
        originals=json.loads((study_path.parent/'original_guards.json').read_text())
        original_voice=json.loads((study_path.parent/'original_voice.json').read_text())
        registry={name:digest(ROOT/name) for name in list(freeze['source_files_sha256'])+[
            'experiments/probe_ticodec_band.py','docs/codec-ticodec-band.md','experiments/codec_profiles_report.py']}
        if registry['experiments/probe_ticodec_band.py']!=IMPLEMENTATION_SHA256:
            raise ValueError('Loaded band generator changed')
        watched={str(p):digest(p) for p in (study_path,ti_path,frame_path)}
        watched.update({str(study_path.parent/name):digest(study_path.parent/name)
            for name in ('original_guards.json','original_voice.json')})
        directory.mkdir(parents=True);manifest=directory/'generation.json'
        report={'kind':'Spent-source Native/TiCodec middle-band generation panel','complete':False,
            'completed_study_sha256':digest(study_path),'ticodec_generation_sha256':digest(ti_path),
            'frame_generation_sha256':digest(frame_path),'code_files_sha256':registry,
            'planned_sources':len(cases),'planned_treatments':len(cases)*len(BANDS),'cases':cases,
            'stems_quantized_before_mixing':True,'no_original_waveform_in_mixture':True,
            'new_http_requests':0,'new_tts_requests':0,'fresh_efficacy_sources':0,
            'reference_text_used_for_generation_or_repair':False,'detector_checked':False,
            'independent_quality_evaluated':False,'human_listening_approval':False,
            'physical_erasure_proven':False,'started_at':datetime.now(timezone.utc).isoformat()}
        save(manifest,report)
        def unchanged():
            if (any(digest(ROOT/name)!=value for name,value in registry.items()) or
                    any(digest(Path(name))!=value for name,value in watched.items())):
                raise ValueError('Generation code or completed input evidence changed')
        try:
            codec=(codec_factory or NativeCodec)()
            recognizer=StudyRecognizer(study_path.parent/'recognition/originals',directory/'recognition/processed',
                source_digests={c['source_sha256']:c['recognition_cache_sha256'] for c in originals['clips']})
            if recognizer.identity!=originals['configuration_sha256']:
                raise ValueError('Recognizer configuration differs from original-only calibration')
            report.update(native_codec=codec.metadata,ticodec=ti['codec'],recognizers=recognizer.metadata,
                recognition_configuration_sha256=recognizer.identity)
            save(manifest,report)
            for case in cases:
                unchanged();source=ROOT/case['source'];folder=directory/case['id'];folder.mkdir()
                case['native_baseline_status']='processing';save(manifest,report)
                try:
                    native,rate,baseline=native_stem(codec,source,folder/'native.flac',case['expected_native_sha256'])
                except NativeBaselineMismatch:
                    case.update(native_baseline_status='mismatch',actual_native_sha256=digest(folder/'native.flac'))
                    save(manifest,report);raise
                case.update(native_baseline_status='exact_match',native_baseline=baseline)
                donor,donor_rate=sf.read(case['ticodec_file'],dtype='float32')
                if donor_rate!=rate or digest(Path(case['ticodec_file']))!=case['ticodec_sha256']:
                    raise ValueError('TiCodec stem changed')
                before=recognizer.transcribe(source)
                for treatment in case['treatments']:
                    treatment['status']='processing';save(manifest,report)
                    output=folder/(treatment['method']+'.flac')
                    mixed,gain=mix_reconstructions(native,donor,rate,*treatment['band_hz'])
                    quantize(output,mixed,rate);checksum=digest(output)
                    treatment.update(output_file=str(output),output_sha256=checksum,gain=gain)
                    after=recognizer.transcribe_candidate(output,before)
                    speech=assess(before,after)
                    treatment.update(speech_assessment=speech,
                        recognition_cache_sha256=digest(directory/'recognition/processed'/(checksum+'.json')))
                    if not speech['content_guard_passed']:
                        treatment['status']='content_rejected'
                    else:
                        quality=runtime_voice_decision(directory,case['id'],case['source_sha256'],
                            case['detected_language'],original_voice,source,output)
                        treatment.update(runtime_quality_guard=quality,
                            status='processing_eligible' if quality['passed'] else 'runtime_quality_rejected')
                    if digest(output)!=checksum or digest(source)!=case['source_sha256']:
                        raise ValueError('Assessed candidate or source changed')
                    unchanged();save(manifest,report)
                    print('TICODEC_BAND_GENERATED',case['id'],treatment['method'],treatment['status'],flush=True)
            report.update(complete=True,completed_at=datetime.now(timezone.utc).isoformat());save(manifest,report)
        except BaseException as error:
            report.update(error_type=type(error).__name__,stopped_at=datetime.now(timezone.utc).isoformat())
            save(manifest,report);raise
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-study',type=Path,required=True)
    parser.add_argument('--ticodec-generation',type=Path,required=True)
    parser.add_argument('--frame-generation',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();run(args.completed_study,args.ticodec_generation,args.frame_generation,args.directory)
