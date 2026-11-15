"""Validate every latent treatment and exact control without running inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest
from codec_content import assess
from experiments.codec_profile_cache import read_recognition
from experiments.ticodec_generation_report import export as original_export
from experiments.ticodec_latents import MODES


def load(path):
    return json.loads(Path(path).read_text())


def validated_inventory(directory):
    import soundfile as sf
    directory = Path(directory).resolve()
    panel = directory/'ticodec_latent_development'
    study_path = directory/'codec_profiles_fresh/curated_complete_report.json'
    original_path = directory/'ticodec_development/generation.json'
    manifest = panel/'generation.json'
    measured, study, original = load(manifest), load(study_path), load(original_path)
    manifest_sha256 = digest(manifest)
    expected = [c for c in study['cases'] if c['marked_provenance'] and not c['qualified_marked_transition']]
    if (measured.get('complete') is not True or measured.get('error_type') or
            measured['completed_study_sha256'] != digest(study_path) or
            measured['original_ticodec_generation_sha256'] != digest(original_path) or
            [c['id'] for c in measured['cases']] != [c['id'] for c in expected] or
            measured['planned_spent_sources'] != len(expected) or
            measured['planned_treatments'] != 3*len(expected) or
            measured['planned_exact_file_controls'] != len(expected)):
        raise ValueError('Require the complete latent panel and every spent marked failure')
    original_export(directory)
    synthetic = measured['synthetic_control']
    if (synthetic['exact_original_adapter_waveform'] is not True or synthetic['maximum_absolute_error'] != 0 or
            any(digest(ROOT/name) != value for name, value in measured['code_files_sha256'].items())):
        raise ValueError('The quantized implementation control or recorded source changed')
    identity = measured['recognition_configuration_sha256']
    if identity != original['recognition_configuration_sha256']:
        raise ValueError('Keep the original processing recognizer configuration')
    assessment = {}

    def retain(path, checksum=None):
        path = Path(path).resolve()
        actual = digest(path)
        if checksum is not None and actual != checksum:
            raise ValueError('A measured source, control or assessment changed')
        assessment[str(path)] = actual

    retain(original_path, measured['original_ticodec_generation_sha256'])
    F = study_path.parent
    retain(F/'original_guards.json')
    retain(F/'original_voice.json')
    original_voice = load(F/'original_voice.json')
    originals = {c['id']: c for c in original['cases']}
    modes = list(MODES[1:])
    cases, sources, treatments = [], [], []
    for case, old in zip(measured['cases'], expected):
        prior = originals[case['id']]
        source = (ROOT/case['source']).resolve()
        first = sf.info(source)
        grid = {'samples': first.frames, 'sample_rate': first.samplerate, 'channels': first.channels}
        control = Path(case['control_file']).resolve()
        if (case['source_sha256'] != old['source_sha256'] or grid != old['source_format'] or
                case['language'] != old['language'] or case['control_status'] != 'exact_match' or
                case['control_expected_sha256'] != prior['output_sha256'] or
                case['control_actual_sha256'] != prior['output_sha256'] or
                not control.is_relative_to(panel)):
            raise ValueError('Retain every exact whole-file quantized control and original sample grid')
        retain(source, old['source_sha256'])
        retain(control, prior['output_sha256'])
        control_info = sf.info(control)
        if (control_info.frames, control_info.samplerate, control_info.channels) != (
                first.frames, first.samplerate, first.channels):
            raise ValueError('The quantized control changed the original sample grid')
        if [(t['mode'], t['local_quantized'], t['global_quantized']) for t in case['treatments']] != modes:
            raise ValueError('Keep all three latent cells in their declared order')
        original_cache = F/'recognition/originals'/(old['source_sha256']+'.json')
        retain(original_cache, old['original_guard_calibration']['recognition_cache_sha256'])
        before = read_recognition(F/'recognition/originals', old['source_sha256'], identity)
        sources.append({'id': old['id'], 'file': str(source), 'sha256': old['source_sha256'], 'format': grid})
        rows = []
        for item in case['treatments']:
            output = Path(item['output_file']).resolve()
            checksum = item['output_sha256']
            second = sf.info(output)
            if (not output.is_relative_to(panel) or digest(output) != checksum or
                    (second.frames, second.samplerate, second.channels) != (
                        first.frames, first.samplerate, first.channels)):
                raise ValueError('Every treatment must retain its unchanged complete waveform')
            cache = panel/'recognition/processed'/(checksum+'.json')
            retain(cache, item['recognition_cache_sha256'])
            after = read_recognition(panel/'recognition/processed', checksum, identity)
            speech = assess(before, after)
            if speech != item['speech_assessment']:
                raise ValueError('Speech assessment differs from its actual recognition cache')
            status, quality = item['status'], item.get('runtime_quality_guard')
            if status == 'content_rejected':
                if speech['content_guard_passed'] or quality is not None:
                    raise ValueError('A speech rejection has inconsistent evidence')
            elif status in ('processing_eligible', 'runtime_quality_rejected'):
                receipt_path = panel/'runtime_voice'/case['id']/(checksum+'.json')
                retain(receipt_path, quality['receipt_sha256'])
                receipt = load(receipt_path)
                if len(receipt['clips']) != 1:
                    raise ValueError('Keep one voice observation per actual file')
                clip = receipt['clips'][0]
                original_clip = next(c for c in original_voice['clips'] if c['id'] == case['id'])
                passing = clip['speaker_minimum'] >= .95 and clip['dnsmos_delta']['overall'] >= -.2
                if (not speech['content_guard_passed'] or receipt.get('complete') is not True or
                        receipt['metric'] != original_voice['metric'] or clip['status'] != 'scored' or
                        clip['original_dnsmos'] != original_clip['original_dnsmos'] or
                        clip['source_sha256'] != old['source_sha256'] or clip['output_sha256'] != checksum or
                        quality['source_sha256'] != old['source_sha256'] or quality['candidate_sha256'] != checksum or
                        quality['speaker_minimum'] != clip['speaker_minimum'] or
                        quality['dnsmos_overall_delta'] != clip['dnsmos_delta']['overall'] or
                        quality['speaker_minimum_required'] != .95 or quality['dnsmos_overall_delta_required'] != -.2 or
                        type(quality['passed']) is not bool or quality['passed'] != passing or
                        passing != (status == 'processing_eligible')):
                    raise ValueError('Voice receipts or frozen quality limits changed')
            else:
                raise ValueError('A complete panel contains unfinished or unknown work')
            rows.append({k: item[k] for k in ('mode', 'local_quantized', 'global_quantized', 'status',
                         'output_sha256', 'gain', 'speech_assessment', 'recognition_cache_sha256')})
            rows[-1]['runtime_quality_guard'] = quality
            treatments.append({'id': case['id'], 'method': item['mode'], 'sha256': checksum,
                'file': str(output), 'processing_status': status, 'detector_eligible': status == 'processing_eligible',
                'endpoint_phase_status': 'pending' if status == 'processing_eligible' else 'processing_rejected'})
        cases.append({'id': case['id'], 'language': old['language'], 'source_sha256': old['source_sha256'],
            'source_format': grid, 'source_model': old['model'], 'voice_id': old['voice'],
            'control_status': case['control_status'], 'control_sha256': prior['output_sha256'], 'treatments': rows})
    if digest(manifest) != manifest_sha256:
        raise ValueError('The generation panel changed during validation')
    public = {'kind': 'Spent-source TiCodec local/global latent generation results only',
        'generation_sha256': manifest_sha256, 'formatter_source_sha256': digest(Path(__file__)),
        'completed_study_sha256': measured['completed_study_sha256'],
        'original_ticodec_generation_sha256': measured['original_ticodec_generation_sha256'],
        'code_files_sha256': measured['code_files_sha256'], 'model': measured['codec'],
        'synthetic_control': synthetic, 'recognition_configuration_sha256': identity,
        'planned_spent_sources': len(cases), 'planned_treatments': len(treatments),
        'exact_quantized_controls': len(cases), 'cases': cases,
        'operational_status_counts': dict(Counter(t['processing_status'] for t in treatments)),
        'generator_new_http_requests': 0, 'new_tts_requests': 0, 'fresh_efficacy_sources': 0,
        'independent_quality_evaluated_by_generation': False, 'detector_evaluated_by_generation': False,
        'completed_study_result_changed': False, 'human_listening_approval': False, 'physical_erasure_proven': False,
        'limits': ['All recordings are spent failures of the completed 19/44 study.',
                   'The 25 exact quantized controls are paired implementation checks, not new sources.',
                   'Processing eligibility does not establish detector clearance or reference-qualified quality.',
                   'Continuous decoder inputs are outside the original quantized codec interface and nominal bitrate.']}
    private = {'sources': sources, 'treatments': treatments, 'assessment_files_sha256': assessment}
    return public, private


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new report; existing evidence is not overwritten')
    public, _ = validated_inventory(args.directory)
    args.output.write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
    print('TICODEC_LATENT_GENERATION_EXPORTED', public['operational_status_counts'])
