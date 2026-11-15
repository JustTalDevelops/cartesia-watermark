"""Validate and export the complete X-Codec2 development panel, without models."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest
from codec_content import assess
from experiments.codec_profile_cache import read_recognition


def load(path):
    return json.loads(path.read_text())


def validate_parity(case):
    controls = case['channel_parity_controls']
    if [row['channel'] for row in controls] != list(range(case['source_format']['channels'])):
        raise ValueError('Require a separate actual full-model control for every complete channel')
    for row in controls:
        if any(row.get(key) is not True for key in
               ('code_indices_exact', 'decoded_waveform_exact', 'rng_advance_exact')):
            raise ValueError('Codes, decoded samples and RNG advancement must all match exactly')
        if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] <= 0
               for key in ('full_seconds', 'prefix_seconds')):
            raise ValueError('Retain actual finite positive component timings')


def validate_completion(measured, declared, study):
    expected = [c for c in study['cases'] if c['marked_provenance'] and not c['qualified_marked_transition']]
    ids = [c['id'] for c in expected]
    if (measured.get('complete') is not True or measured.get('error_type') or
            [c['id'] for c in measured['cases']] != ids or
            [c['id'] for c in declared['cases']] != ids or
            len(set(ids)) != len(ids) or len(ids) != 25 or
            any(measured.get(key) != len(ids) or declared.get(key) != len(ids)
                for key in ('planned_sources', 'planned_treatments'))):
        raise ValueError('Require the complete fixed panel, including every original failure')
    if (measured.get('new_tts_requests') != 0 or measured.get('new_http_requests') != 0 or
            measured.get('fresh_efficacy_sources') != 0 or
            measured.get('reference_text_used_for_generation_or_repair') is not False or
            measured.get('independent_quality_evaluated') is not False or
            measured.get('detector_checked') is not False):
        raise ValueError('Generation must remain separate from detection and independent evaluation')
    for case in measured['cases']:
        validate_parity(case)
        if case['status'] not in ('content_rejected', 'processing_eligible', 'runtime_quality_rejected'):
            raise ValueError('A complete panel cannot retain unfinished or unknown rows')
    return expected


def export(directory):
    import soundfile as sf
    directory = Path(directory).resolve(); panel = directory/'xcodec2_development'
    original = directory/'codec_profiles_fresh'
    measured = load(panel/'generation.json'); declared = load(panel/'declaration.json')
    study_path = original/'curated_complete_report.json'; study = load(study_path)
    if (digest(study_path) != '777355dff6ed90ec0beec56558b0c3a4fc974abb88c38993d95d554c6040cd4b' or
            measured['declaration_sha256'] != digest(panel/'declaration.json') or
            measured['code_files_sha256'] != declared['code_files_sha256'] or
            measured['input_files_sha256'] != declared['input_files_sha256']):
        raise ValueError('Keep the original study and the prospectively recorded generation inputs')
    for path, sha in {**measured['code_files_sha256'], **measured['input_files_sha256']}.items():
        if digest(Path(path)) != sha:
            raise ValueError('A frozen generation input or implementation changed')
    expected = validate_completion(measured, declared, study)
    identity = measured['recognition_configuration_sha256']; voice = load(original/'original_voice.json')
    rows = []
    for case, prior, planned in zip(measured['cases'], expected, declared['cases']):
        source, output = Path(case['source']), Path(case['output_file'])
        first, second = sf.info(source), sf.info(output)
        grid = {'samples': first.frames, 'sample_rate': first.samplerate, 'channels': first.channels}
        if (case['source_sha256'] != prior['source_sha256'] or
                any(case[key] != planned[key] for key in ('source', 'source_sha256', 'source_format')) or
                digest(source) != case['source_sha256'] or digest(output) != case['output_sha256'] or
                not output.resolve().is_relative_to(panel) or grid != case['source_format'] or
                grid != prior['source_format'] or
                (first.frames, first.samplerate, first.channels) !=
                (second.frames, second.samplerate, second.channels)):
            raise ValueError('Bind the exact complete input and generated file to the original sample grid')
        before_path = original/'recognition/originals'/(case['source_sha256']+'.json')
        after_path = panel/'recognition/processed'/(case['output_sha256']+'.json')
        if (digest(before_path) != prior['original_guard_calibration']['recognition_cache_sha256'] or
                digest(after_path) != case['recognition_cache_sha256']):
            raise ValueError('Recognition receipts changed')
        speech = assess(read_recognition(before_path.parent, case['source_sha256'], identity),
                        read_recognition(after_path.parent, case['output_sha256'], identity))
        if speech != case['speech_assessment']:
            raise ValueError('Recorded speech gate differs from the frozen recognizer caches')
        quality = case.get('runtime_quality_guard')
        if case['status'] == 'content_rejected':
            if speech['content_guard_passed'] or quality is not None:
                raise ValueError('A speech rejection cannot imply a completed voice assessment')
        else:
            receipt_path = panel/'runtime_voice'/case['id']/(case['output_sha256']+'.json')
            receipt = load(receipt_path); clip = receipt['clips'][0]
            before = next(c for c in voice['clips'] if c['id'] == case['id'])
            if (not speech['content_guard_passed'] or digest(receipt_path) != quality['receipt_sha256'] or
                    receipt.get('complete') is not True or receipt['metric'] != voice['metric'] or
                    clip['original_dnsmos'] != before['original_dnsmos'] or
                    clip['source_sha256'] != case['source_sha256'] or
                    clip['output_sha256'] != case['output_sha256'] or
                    clip['speaker_minimum'] != quality['speaker_minimum'] or
                    clip['dnsmos_delta']['overall'] != quality['dnsmos_overall_delta']):
                raise ValueError('Operational voice assessment differs from its actual receipt')
            passing = clip['speaker_minimum'] >= .95 and clip['dnsmos_delta']['overall'] >= -.2
            if (quality.get('passed') is not passing or
                    quality['speaker_minimum_required'] != .95 or
                    quality['dnsmos_overall_delta_required'] != -.2 or
                    passing != (case['status'] == 'processing_eligible')):
                raise ValueError('Keep both original voice limits and their actual decisions')
        row = {key: case[key] for key in ('id', 'language', 'detected_language', 'source_sha256',
               'output_sha256', 'source_format', 'status', 'gain', 'channel_parity_controls',
               'codec_with_parity_seconds', 'speech_assessment', 'recognition_cache_sha256')}
        row.update(source_model=prior['model'], voice_id=prior['voice'],
                   source_recognition_cache_sha256=digest(before_path), runtime_quality_guard=quality)
        rows.append(row)
    # Public provenance contains hashes and portable labels, never local filesystem paths.
    code = {}
    for name, sha in measured['code_files_sha256'].items():
        path = Path(name)
        label = ('private-prototype/'+path.name if path.parent == directory else
                 'transformers/'+path.name if 'site-packages' in path.parts else
                 '/'.join(path.parts[-2:]))
        if label in code: raise ValueError('Provenance labels must remain unambiguous')
        code[label] = sha
    return {'kind': 'Complete spent-source X-Codec2 generation and exact-prefix controls',
        'generation_sha256': digest(panel/'generation.json'),
        'declaration_sha256': digest(panel/'declaration.json'),
        'completed_study_sha256': digest(study_path), 'model': measured['codec'],
        'code_files_sha256': code, 'formatter_source_sha256': digest(Path(__file__)),
        'planned_spent_sources': len(rows), 'planned_treatments': len(rows),
        'operational_status_counts': dict(Counter(c['status'] for c in rows)),
        'recognition_configuration_sha256': identity, 'cases': rows,
        'new_tts_requests': 0, 'new_http_requests': 0, 'fresh_efficacy_sources': 0,
        'independent_quality_evaluated': False, 'detector_checked': False,
        'completed_study_result_changed': False, 'human_listening_approval': False,
        'physical_erasure_proven': False,
        'limits': ['Every recording is a spent failure of the completed 19/44 study.',
                   'Processing eligibility does not establish removal or independent quality.',
                   'Timing pairs measure CPU codec components, excluding ASR, loading, APIs and app overhead.',
                   'The upstream noncommercial model was not trained by this project and is not redistributed.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.directory)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('XCODEC2_GENERATION_EXPORTED', result['operational_status_counts'])
