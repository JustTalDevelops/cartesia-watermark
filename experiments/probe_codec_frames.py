"""Generate the fixed frame-offset panel from a completed study's spent failures.

No detector or TTS requests are made. The completed report is revalidated before
models load. Recognition and voice estimates keep the existing processing rules;
reference text and evaluation scores do not generate or repair a treatment.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest
from experiments.codec_frame_alignment import FrameAlignedCodec, DEVELOPMENT_OFFSETS, IMPLEMENTATION_SHA256

DRIVER_SHA256 = digest(Path(__file__))

PHASES = ('baseline', 'original_guards', 'original_evaluation', 'original_voice',
          'candidates', 'processed_evaluation', 'processed_voice')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


@contextmanager
def without_http():
    import requests
    original = requests.sessions.Session.request
    previous = {key:os.environ.get(key) for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
    def blocked(*args, **kwargs):
        raise RuntimeError('HTTP requests are disabled for the framing generation panel')
    requests.sessions.Session.request = blocked
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    try:
        yield
    finally:
        requests.sessions.Session.request = original
        for key,value in previous.items():
            if value is None:os.environ.pop(key, None)
            else:os.environ[key] = value


def completed_study(path):
    directory = path.parent
    report = json.loads(path.read_text())
    for name in PHASES:
        measured = json.loads((directory/(name+'.json')).read_text())
        if measured.get('complete') is not True or measured.get('error_type'):
            raise ValueError('Finish the current study before probing frames: '+name)
    if (report.get('kind')!='Frozen prospective language-profile study' or
            report.get('planned_requests')!=48 or len(report.get('cases', []))!=48 or
            report.get('marked_provenance_requests')!=44):
        raise ValueError('Use the completed, curated 48-case profile study')
    freeze_path = ROOT/'results/codec_profiles_reliability.freeze.json'
    if digest(freeze_path)!=report['freeze_sha256']:
        raise ValueError('Completed report belongs to another frozen study')
    from experiments.codec_profiles_report import export
    if export(freeze_path, directory)!=report:
        raise ValueError('Completed study report differs from its actual observations')
    return report, json.loads((directory/'evaluation.plan.json').read_text()), freeze_path


def failed_marked_cases(report):
    return [row for row in report['cases']
            if row['marked_provenance'] and not row['qualified_marked_transition']]


def historical_native_sha(record):
    observed = {a['model']['native_codec_sha256'] for a in record['attempts']
                if a.get('model', {}).get('native_codec_sha256')}
    if len(observed)>1:
        raise ValueError('Historical native baselines disagree')
    return next(iter(observed), None)


def generate_case(runtime, base, source, row, directory, voice_guard, checkpoint):
    """Retain all four treatments; a failed baseline stops the panel."""
    import soundfile as sf
    directory.mkdir()
    for treatment in row['treatments']:
        offset = treatment['offset_native_samples']
        treatment['status'] = 'processing'
        checkpoint()
        if digest(source)!=row['source_sha256']:
            raise ValueError('Spent source changed before framing')
        runtime.clear_job()
        runtime.codec = FrameAlignedCodec(base, offset)
        raw = directory/f'offset-{offset}.wav'
        metadata = runtime.process_file(source, raw, 'codec-native', directory/f'offset-{offset}.json')
        treatment['model'] = metadata
        if digest(source)!=row['source_sha256']:
            raise ValueError('Spent source changed during framing')
        checksum = metadata['candidate_flac_sha256']
        if offset==0:
            expected = row['historical_native_sha256']
            treatment['historical_baseline_comparison'] = ('unpaired_baseline' if expected is None else
                'exact_match' if checksum==expected else 'mismatch')
            if expected is not None and checksum!=expected:
                treatment['status'] = 'baseline_mismatch'
                checkpoint()
                raise ValueError('Zero-offset native output differs from its historical bytes')
        if not metadata['content_guard_passed']:
            treatment['status'] = 'content_rejected'
            checkpoint()
            continue
        output = directory/f'offset-{offset}.flac'
        audio, rate = sf.read(raw, dtype='float32')
        with output.open('xb') as handle:
            sf.write(handle, audio, rate, format='FLAC', subtype='PCM_24')
        if digest(output)!=checksum:
            raise ValueError('Framing output differs from the exact assessed encoded file')
        if digest(source)!=row['source_sha256']:
            raise ValueError('Spent source changed during framing')
        treatment.update(output_sha256=checksum, output_file=str(output))
        quality = voice_guard(source, output, row)
        if type(quality.get('passed')) is not bool:
            raise ValueError('Voice gate did not return its required decision')
        treatment.update(runtime_quality_guard=quality,
            status='processing_eligible' if quality['passed'] else 'runtime_quality_rejected')
        raw.unlink()
        checkpoint()


def run(completed_path, directory, codec_factory=None):
    completed_path, directory = Path(completed_path), Path(directory)
    if directory.exists() or directory.resolve().is_relative_to(completed_path.parent.resolve()):
        raise ValueError('Choose a new probe directory outside the completed study')
    with without_http():
        completed, plan, freeze_path = completed_study(completed_path)
        study_directory = completed_path.parent
        freeze = json.loads(freeze_path.read_text())
        inputs = {item['id']:item for item in plan['items']}
        old = json.loads((study_directory/'candidates.json').read_text())
        old_cases = {row['id']:row for row in old['cases']}
        originals = json.loads((study_directory/'original_guards.json').read_text())
        original_voice = json.loads((study_directory/'original_voice.json').read_text())
        guards = {row['id']:row for row in originals['clips']}
        registry_names = list(freeze['source_files_sha256'])+[
            'experiments/codec_frame_alignment.py', 'experiments/probe_codec_frames.py',
            'experiments/codec_profiles_report.py']
        registry = {name:digest(ROOT/name) for name in registry_names}
        if (any(registry[name]!=checksum for name,checksum in freeze['source_files_sha256'].items()) or
                registry['experiments/codec_frame_alignment.py']!=IMPLEMENTATION_SHA256 or
                registry['experiments/probe_codec_frames.py']!=DRIVER_SHA256):
            raise ValueError('Loaded probe or processing code differs from its recorded files')
        cases = []
        for failed in failed_marked_cases(completed):
            item = inputs[failed['id']]
            cases.append({'id':item['id'], 'source':item['source'], 'source_sha256':item['sha256'],
                'requested_language':item['language'], 'detected_language':guards[item['id']]['detected_language'],
                'original_detected':failed['original_detected'],
                'historical_native_sha256':historical_native_sha(old_cases[item['id']]),
                'treatments':[{'offset_native_samples':offset, 'status':'planned'} for offset in DEVELOPMENT_OFFSETS]})
        report = {'kind':'Spent-source codec frame-alignment generation panel',
            'completed_study_sha256':digest(completed_path), 'freeze_sha256':digest(freeze_path),
            'code_files_sha256':registry, 'offsets_native_samples':list(DEVELOPMENT_OFFSETS),
            'codec_bandwidth_kbps':12., 'planned_sources':len(cases),
            'planned_treatments':len(DEVELOPMENT_OFFSETS)*len(cases),
            'cases':cases, 'new_tts_requests':0, 'new_http_requests':0, 'fresh_efficacy_sources':0,
            'reference_text_used_for_generation_or_repair':False, 'independent_quality_evaluated':False,
            'detector_phase_declared':False, 'human_listening_approval':False, 'physical_erasure_proven':False,
            'complete':False, 'started_at':datetime.now(timezone.utc).isoformat()}
        directory.mkdir(parents=True)
        manifest = directory/'generation.json'
        checkpoint = lambda:save(manifest, report)
        checkpoint()
        runtime = None
        try:
            if cases:
                from codec_inference import NativeCodec
                from reconstruct_codec import CodecRuntime
                from experiments.codec_profile_cache import StudyRecognizer
                from experiments.run_codec_profiles import runtime_voice_decision
                base = (codec_factory or NativeCodec)()
                source_digests = {row['source_sha256']:row['recognition_cache_sha256'] for row in originals['clips']}
                recognizer = StudyRecognizer(study_directory/'recognition/originals', directory/'recognition/processed',
                    source_digests=source_digests)
                if recognizer.identity!=originals['configuration_sha256']:
                    raise ValueError('Processing recognizer differs from the source-only calibration')
                runtime = CodecRuntime(codec=base, recognizer=recognizer)
                report.update(native_codec=base.metadata, recognizers=recognizer.metadata,
                    recognition_configuration_sha256=recognizer.identity)
                for row in cases:
                    if any(digest(ROOT/name)!=checksum for name,checksum in registry.items()):
                        raise ValueError('Probe or frozen processing code changed')
                    source = ROOT/row['source']
                    def voice_guard(original, candidate, case):
                        return runtime_voice_decision(directory, case['id'], case['source_sha256'],
                            case['detected_language'], original_voice, original, candidate)
                    generate_case(runtime, base, source, row, directory/row['id'], voice_guard, checkpoint)
                    if any(digest(ROOT/name)!=checksum for name,checksum in registry.items()):
                        raise ValueError('Probe or frozen processing code changed during generation')
            report.update(complete=True, completed_at=datetime.now(timezone.utc).isoformat())
            checkpoint()
            print('FRAME_PANEL_GENERATED', len(cases), 'spent sources; zero detector/TTS requests', flush=True)
            return report
        except BaseException as error:
            report.update(error_type=type(error).__name__, stopped_at=datetime.now(timezone.utc).isoformat())
            checkpoint()
            raise
        finally:
            if runtime is not None:runtime.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-study', type=Path, required=True,
        help='Validated complete report in the original study directory')
    parser.add_argument('--directory', type=Path, required=True, help='New development-output directory')
    arguments = parser.parse_args()
    run(arguments.completed_study, arguments.directory)
