"""Opt-in whole-file Cartesia verification using the frozen language profiles.

Uploads the complete original and candidates to Cartesia. The upload app does
not invoke this CLI. Local recognizer/voice estimates and current detector
negatives do not establish unchanged speech or physical watermark erasure.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
_requested_interval = float(os.environ.get('DETECT_INTERVAL','80'))
os.environ['DETECT_PROXY']=''
os.environ['DETECT_INTERVAL']=str(max(80.,_requested_interval))
import cartesia
from codec_audio import digest,validate_audio
from codec_content import ContentRecognizer
from codec_profiles import method_order,PROFILE_VERSION
from reconstruct_codec import CodecRuntime
from verify_reconstruction import reconstruct_verified,NotCleared,publish_exact,remove_own_output,save_report
from experiments.codec_profile_cache import StudyRecognizer,configuration,store_recognition
from experiments.run_codec_profiles import RuntimeAdapter,runtime_voice_decision

# The study helper sets its frozen interval on import; this public wrapper can
# honor a caller's slower setting while retaining the same minimum.
cartesia.MIN_INTERVAL=max(80.,_requested_interval)


def verify(source,output,audit_path,detector=None,recognizer_factory=None):
    import soundfile as sf
    source,output,audit_path=map(Path,(source,output,audit_path))
    if recognizer_factory is not None and not callable(recognizer_factory):
        raise ValueError('A recognizer factory must be callable')
    if (source.resolve() in (output.resolve(),audit_path.resolve()) or output.resolve()==audit_path.resolve() or
            output.exists() or audit_path.exists() or output.suffix.lower()!='.flac'):
        raise ValueError('Choose new FLAC and audit paths separate from the source')
    audio,rate=sf.read(source,dtype='float32');validate_audio(audio,rate)
    if source.stat().st_size>32_000_000:
        raise ValueError('The complete original exceeds the detector upload limit')
    if detector is None:
        if not cartesia._env('CARTESIA_API_KEY'):
            raise ValueError('Set CARTESIA_API_KEY before uploading to Cartesia')
        detector=cartesia.detect
    output.parent.mkdir(parents=True,exist_ok=True);audit_path.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'checking_original','profile_version':PROFILE_VERSION,'source_sha256':digest(source),
            'sample_rate':rate,'samples':len(audio),'channels':1 if audio.ndim==1 else audio.shape[1],
            'checks':[],'reference_text_used':False,'independent_reference_quality_verified':False,
            'quality_verified':False,'physical_erasure_proven':False,'uploads_original_and_candidates':True}
    with audit_path.open('x') as handle:json.dump(report,handle,indent=2)
    published_identity=None

    def save():
        save_report(audit_path,report)

    def check(path,role,ordinal):
        checksum=digest(path);status,response=detector(path)
        if digest(path)!=checksum or digest(source)!=report['source_sha256']:
            raise RuntimeError('Original or candidate changed during verification')
        entry={'role':role,'ordinal':ordinal,'sha256':checksum,'http_status':status,
               'checked_at':datetime.now(timezone.utc).isoformat()}
        if isinstance(response,dict) and type(response.get('detected')) is bool:entry['detected']=response['detected']
        report['checks'].append(entry);save()
        if status!=200 or 'detected' not in entry:raise RuntimeError('Detector unavailable; no output published')
        return status,response

    def repeated(path,role,expected=None):
        first=check(path,role,1)[1]['detected'];second=check(path,role,2)[1]['detected']
        if first!=second or expected is not None and first!=expected:raise RuntimeError('Repeated detector/control responses conflict')
        return first

    try:
        if not repeated(source,'opening_original'):
            report['status']='original_not_detected';save();return report
        with tempfile.TemporaryDirectory(prefix='.codec-profile-',dir=output.parent) as temporary:
            directory=Path(temporary)
            real=(ContentRecognizer if recognizer_factory is None else recognizer_factory)()
            before=real.transcribe(source)
            report['processing_recognizers']=real.metadata
            if digest(source)!=report['source_sha256']:raise RuntimeError('Original changed during recognition')
            if not before['supported']:
                report.update(status='unsupported_language',detected_language=before['language']);save();return report
            identity=configuration(real.metadata)
            original=store_recognition(directory/'recognition/originals',report['source_sha256'],identity,before)
            rec=StudyRecognizer(directory/'recognition/originals',directory/'recognition/processed',real,
                                source_digests={report['source_sha256']:digest(original)})
            voice_plan=directory/'original_voice.plan.json';voice_output=directory/'original_voice.json'
            voice_plan.write_text(json.dumps({'items':[{'id':'input','source':str(source.resolve()),
                'sha256':report['source_sha256'],'language':before['language'],'text':''}]},indent=2)+'\n')
            with (directory/'original_voice.log').open('x') as log:
                subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'experiments/score_codec_study.py'),
                    '--plan',str(voice_plan),'--metric','voice','--originals-only','--output',str(voice_output),
                    '--cache',str(directory/'scores/cache')],stdout=log,stderr=subprocess.STDOUT,check=True)
            voice=json.loads(voice_output.read_text())
            if not voice['complete']:raise RuntimeError('Original voice preflight incomplete')
            orders=method_order(before['language'],len(audio)/rate)
            report.update(status='processing',detected_language=before['language'],method_order=list(orders),
                          original_recognition_reused=True,original_voice_metric=voice['metric'])
            save();runtime=CodecRuntime(recognizer=rec);staged=directory/'checked.flac';processing=directory/'processing.json'

            def candidate_detector(path):
                attempt=json.loads(processing.read_text())['attempts'][-1]
                role='processed_'+attempt['preset'];ordinal=attempt['checks'][-1]['ordinal']
                status,response=check(path,role,ordinal)
                if ordinal>1 and response['detected']!=attempt['checks'][0]['detected']:
                    raise RuntimeError('Candidate repeated detector responses conflict')
                if response['detected'] and check(path,role+'_positive_confirmation',2)[1]['detected'] is not True:
                    raise RuntimeError('Candidate positive confirmation conflicts')
                return status,response

            try:
                selected=reconstruct_verified(source,staged,processing,orders,worker=RuntimeAdapter(runtime),
                    detector=candidate_detector,confirmations=2,skip_duplicate_candidates=True,
                    runtime_quality_guard=lambda candidate:runtime_voice_decision(directory,'input',report['source_sha256'],
                        before['language'],voice,source.resolve(),candidate))
            except NotCleared:
                report.update(status='not_cleared',processing=json.loads(processing.read_text()));save();return report
            finally:runtime.close()
            repeated(source,'closing_original',True)
            published_identity=publish_exact(staged,output,selected['output_sha256'])
            report.update(status='verified_negative',output_sha256=digest(output),selected_preset=selected['selected_preset'],
                          negative_checks=2,processing=selected,
                          meaning='Current whole-file detector negatives with local speech/voice gates; no independent reference-text or human quality approval')
            save();return report
    except Exception as error:
        if published_identity is not None:remove_own_output(output,published_identity)
        report.update(status='verification_failed',error_type=type(error).__name__);save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path,help='New FLAC, published only after complete source controls and output checks')
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--experimental-packed-large',action='store_true',
                        help='Experimental Hindi/Tamil recognizer weight packing; outside the frozen study')
    args=parser.parse_args();protocol=sys.stdout
    factory=None
    if args.experimental_packed_large:
        from codec_packed_fp16 import PackedLargeRecognizer
        factory=PackedLargeRecognizer
    with redirect_stdout(sys.stderr):
        report=verify(args.input,args.output,args.audit or args.output.with_suffix('.audit.json'),recognizer_factory=factory)
    print(json.dumps(report),file=protocol)
    raise SystemExit(0 if report['status']=='verified_negative' else 2)
