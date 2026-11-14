"""Export checked metadata for spent seed and neural-codec development.

No recordings, recognized text or speaker embeddings are exported. Passing
variants of one recording never increase the independent source count.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.speech_repair_development_report import followup_quality


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def criteria(row):
    q = row['quality_estimates']
    a, s, m, v = [q[n] for n in ('asr_turbo', 'asr', 'mms', 'voice_quality')]
    return {'turbo_paired_disagreement': a['asr_disagreement'] <= .03+1e-12,
            'turbo_reference_delta': a['reference_error_delta'] <= .03+1e-12,
            'small_reference_delta': s['reference_error_delta'] <= .03+1e-12,
            'mms_reference_delta': m['reference_error_delta'] <= .03+1e-12,
            'mms_paired_disagreement': m['paired_disagreement'] <= .03+1e-12,
            'voice_minimum': v['speaker_minimum'] >= .95-1e-12,
            'mos_delta': v['dnsmos_delta']['overall'] >= -.2-1e-12}


def checked_component(directory, stem, expected_outputs, expected_http):
    folder = directory/stem
    rows = followup_quality(directory, stem)
    require(len(rows) == expected_outputs and len({r['source_sha256'] for r in rows}) == 2,
            'Unexpected development output/source count')
    protocol, api = load(folder/'api.protocol.json'), load(folder/'api.json')
    require(api['complete'] is True and api['protocol_sha256'] == digest(folder/'api.protocol.json') and
            len(api['http_attempts']) == len(api['clips']) == expected_http and
            protocol['plan_sha256'] == digest(directory/(stem+'.plan.json')) and
            protocol['render_sha256'] == digest(directory/(stem+'.render.json')),
            'Incomplete or mismatched codec endpoint observations')
    for name, expected in protocol['quality_reports_sha256'].items():
        require(digest(directory/(stem+'.'+name+'.json')) == expected, 'Codec quality changed after qualification')
    selected = []
    for row in rows:
        q = criteria(row)
        require(q == protocol['quality_qualification_before_detector_labels'][row['id']],
                'Codec qualification does not match its fixed criteria')
        row['estimated_quality_criteria'] = q
        observations = [c for c in api['clips'] if c['id'] == row['id']]
        if all(q.values()):
            selected.append(row['id'])
            require(sorted(c['role'] for c in observations) == ['processed_1', 'processed_2'] and
                    all(c['http_status'] == 200 and c['sha256'] == row['output_sha256'] and
                        type(c['detected']) is bool for c in observations), 'Missing exact-file codec checks')
            row['detector_checked'] = True
            row['processed_detected'] = [c['detected'] for c in observations]
            row['status'] = 'checked_negative_twice' if not any(row['processed_detected']) else 'still_detected'
        else:
            require(not observations, 'Quality-rejected codec was silently detector-tested')
            row['status'] = 'estimated_quality_rejected_before_detector'
    require(set(selected) == set(protocol['selected_ids']), 'Codec selection changed after labels')
    controls = [c for c in api['clips'] if c['role'].endswith('positive_control')]
    require(len(controls) == 2 and all(c['http_status'] == 200 and c['detected'] is True for c in controls),
            'Codec service controls failed')
    renders = load(directory/(stem+'.render.json'))
    return {'scope': protocol['scope'], 'source_recordings': 2, 'rendered_variants': len(rows),
            'quality_qualified_variants': len(selected),
            'quality_qualified_sources': len({r['source_sha256'] for r in rows if r['id'] in selected}),
            'protocol_sha256': digest(folder/'api.protocol.json'), 'http_attempts': expected_http,
            'positive_service_controls': 2, 'cases': rows,
            'render_timings': [{k: r[k] for k in ('id', 'elapsed_seconds', 'model_load_seconds') if k in r}
                               for r in renders if 'elapsed_seconds' in r],
            'timing_limits': 'Where recorded, one observation per variant on the same two spent recordings, with model load separate; no general latency claim.'}


def seed_component(directory):
    rows = followup_quality(directory, 'seed_resampling_dev')
    qualification = load(directory/'seed_resampling_dev.qualification.json')
    require(qualification['complete'] is True and qualification['qualified_sources'] == 0 and
            qualification['selected'] == [] and len(rows) == 8 and
            qualification['seed_detector_labels_not_yet_queried'] is True, 'Incomplete seed qualification')
    for row in rows:
        q = criteria(row)
        require(q == qualification['criteria'][row['id']] and not all(q.values()),
                'Seed rejection differs from its quality evidence')
        row['estimated_quality_criteria'] = q
    return {'scope': qualification['scope'], 'source_recordings': 2, 'rendered_variants': 8,
            'fixed_seed_order': [0, 1, 2, 3], 'qualified_sources': 0, 'detector_requests': 0,
            'cases': rows, 'qualification_sha256': digest(directory/'seed_resampling_dev.qualification.json')}


def repair_components(directory):
    """Retain every produced repair and each coverage rejection."""
    result = {}
    for stem, count in [('codec_dual_repair_dev', 4), ('codec_half_donor_dev', 6)]:
        rows = followup_quality(directory, stem)
        require(len(rows) == count, 'Repair output count changed')
        for row in rows:
            row['estimated_quality_criteria'] = criteria(row)
            row['status'] = ('estimated_quality_passed_detector_not_queried_in_this_component'
                             if all(criteria(row).values()) else 'estimated_quality_rejected')
        result[stem] = {'source_recordings': 2, 'rendered_variants': count, 'cases': rows,
                        'render_protocol_sha256': digest(directory/stem/'protocol.json')}
    stem = 'codec_word_boundary_dev'
    document = load(directory/(stem+'.mms.json'))
    require(document['complete'] is True and len(document['clips']) == 2,
            'Incomplete word-boundary review')
    plan = load(directory/(stem+'.plan.json'))
    by_id = {i['id']: i for i in plan['items']}
    for row in document['clips']:
        item = by_id[row['id']]
        require(digest(ROOT/item['source']) == row['source_sha256'] == item['sha256'] and
                digest(ROOT/item['output']) == row['output_sha256'] and row['paired_disagreement'] > .03,
                'Word-boundary rejection is not bound to its actual files')
    result[stem] = {'review': 'MMS only; both Arabic outputs failed paired agreement before further evaluation',
                    'cases': document['clips'], 'detector_requests': 0,
                    'decisions': load(directory/stem/'decisions.json')}
    return result


def quality_rejected_component(directory, stem, expected_outputs, expected_sources):
    rows = followup_quality(directory, stem)
    qualification_path = directory/(stem+'.qualification.json')
    qualification = load(qualification_path)
    require(len(rows) == expected_outputs and len({r['source_sha256'] for r in rows}) == expected_sources and
            qualification['complete'] is True and qualification['selected'] == [] and
            qualification['qualified_sources'] == 0 and qualification['detector_requests'] == 0 and
            not (directory/stem/'api.json').exists(), 'Quality-rejected study has unexpected endpoint evidence')
    for row in rows:
        q = criteria(row)
        require(not all(q.values()) and q == qualification['criteria'][row['id']],
                'Quality rejection does not match actual estimates')
        row['estimated_quality_criteria'] = q
        row['status'] = 'estimated_quality_rejected_before_detector'
    return {'source_recordings': expected_sources, 'rendered_variants': expected_outputs,
            'qualified_sources': 0, 'detector_requests': 0, 'cases': rows,
            'qualification_sha256': digest(qualification_path),
            'render_protocol': load(directory/stem/'protocol.json')}


def arabic_diagnostic(directory):
    folder = directory/'arabic_codec_diagnostic'
    path = folder/'protocol.json'
    protocol, api = load(path), load(folder/'api.json')
    require(api['complete'] is True and api['protocol_sha256'] == digest(path) and
            protocol['quality_qualified'] is False and protocol['usable_removal_success'] is False and
            api['quality_qualified'] is False and api['usable_removal_success'] is False and
            len(api['http_attempts']) == len(api['clips']) == 4, 'Incomplete diagnostic or inflated success claim')
    original = next(i for i in load(directory/'vocos_encodec_development.plan.json')['items']
                    if i['id'] == 'followup_06_ar_sonic-3.6__vocos_encodec_12kbps')
    require(protocol['source_sha256'] == digest(ROOT/original['source']) == original['sha256'] and
            protocol['output_sha256'] == digest(ROOT/original['output']) and
            not all(protocol['quality_criteria'].values()), 'Diagnostic source or quality rejection changed')
    checks = [c for c in api['clips'] if c['role'].startswith('diagnostic_quality_rejected_')]
    require(sorted(c['role'] for c in checks) == ['diagnostic_quality_rejected_1', 'diagnostic_quality_rejected_2'] and
            all(c['sha256'] == protocol['output_sha256'] and c['http_status'] == 200 and
                type(c['detected']) is bool for c in checks), 'Diagnostic used another file')
    controls = [c for c in api['clips'] if c['role'].endswith('positive_control')]
    require(len(controls) == 2 and all(c['http_status'] == 200 and c['detected'] is True and
            c['sha256'] == protocol['canonical_original_sha256'] for c in controls), 'Diagnostic service controls failed')
    return {'scope': protocol['scope'], 'purpose': protocol['purpose'],
            'source_recordings': 1, 'fresh_efficacy_recordings': 0, 'quality_qualified': False,
            'usable_removal_success': False, 'protocol_sha256': digest(path), 'http_attempts': 4,
            'source_sha256': protocol['source_sha256'], 'output_sha256': protocol['output_sha256'],
            'quality_criteria': protocol['quality_criteria'],
            'detected': [c['detected'] for c in checks], 'positive_service_controls': 2,
            'interpretation': 'Separate diagnosis of an unusable quality-rejected output; never an eligible removal success.'}


def band_mix_parity(directory):
    report = load(directory/'codec_band_mix.package_parity.json')
    require(report['complete'] is True and len(report['cases']) == 4 and
            report['source_sha256']['experiments/codec_band_mix.py'] == digest(ROOT/'experiments/codec_band_mix.py'),
            'Band probe parity uses another implementation')
    renders = {r['id']: r for r in load(directory/'codec_vocoder_band_dev.render.json')}
    for row in report['cases']:
        require(row['exact_encoded_flac_parity'] is True and
                row['output_sha256'] == renders[row['id']]['output_sha256'], 'Band probe parity mismatch')
    return report


def additional_arabic_followups(directory):
    report = load(directory/'fastconformer_ar_review/followup_result.json')
    calibration_path = directory/'fastconformer_ar_review/original_calibration.json'
    require(report['complete'] is True and report['source_original_calibration_sha256'] == digest(calibration_path) and
            report['criteria_thresholds_changed'] is False and report['transcripts_exported'] is False and
            len(report['clips']) == len({r['id'] for r in report['clips']}) == 17,
            'Incomplete additional Arabic follow-up review')
    items = {i['id']: i for stem in ['codec_half_donor_dev', 'codec_decoder_cycle_dev', 'dac_fullband_dev',
                                    'codec_partial_repair_dev', 'codec_vocoder_band_dev', 'flow_vocoder_band_dev']
             for i in load(directory/(stem+'.plan.json'))['items']}
    for row in report['clips']:
        item = items[row['id']]
        require(row['calibration_eligible'] is True and row['source_sha256'] == report['source_sha256'] ==
                item['sha256'] == digest(ROOT/item['source']) and row['output_sha256'] == digest(ROOT/item['output']),
                'Additional Arabic follow-up used another file')
    return report


def development_selector(codec, bands):
    """Describe the next candidate, formed after these spent development labels."""
    native = [r for r in codec['cases'] if r['id'].endswith('__vocos_encodec_12kbps')]
    fallback = {r['source_sha256']: r for r in bands['cases'] if r['id'].endswith('_1000_4000')}
    require(len(native) == len(fallback) == 2, 'Missing selector development source')
    decisions = []
    for first in native:
        q = first['quality_estimates']
        content_ok = q['asr_turbo']['asr_disagreement'] <= .03+1e-12 and q['mms']['paired_disagreement'] <= .03+1e-12
        first_negative = first.get('processed_detected') == [False, False]
        selected = first if content_ok and first_negative else fallback[first['source_sha256']]
        require(all(criteria(selected).values()) and selected['processed_detected'] == [False, False],
                'Proposed selector lacks quality-qualified twice-negative evidence')
        decisions.append({'source_sha256': first['source_sha256'], 'native_id': first['id'],
                          'native_content_guard_passed': content_ok, 'fallback_used': selected is not first,
                          'selected_id': selected['id'], 'selected_output_sha256': selected['output_sha256'],
                          'all_seven_estimated_quality_criteria_passed': True, 'selected_detected': [False, False]})
    return {'scope': 'Next candidate assembled from spent development results; no fresh evaluation',
            'rule': 'Try native Vocos/EnCodec 12 kbps first, then the complete 1000–4000 Hz codec/vocoder mixture after content rejection or positive detection. Require paired Turbo and MMS disagreement <=0.03 and two negative checks for the selected file.',
            'source_recordings': 2, 'fresh_efficacy_recordings': 0, 'decisions': decisions,
            'selector_formed_after_development_detector_labels': True,
            'end_to_end_runtime_implemented_and_validated': False, 'human_listening_approval': False,
            'all_selected_files_quality_qualified_and_double_negative': True}


def guarded_workflow(directory):
    """Validate the fixed rule and its unsuccessful Arabic endpoint checks."""
    stem = 'codec_guarded_workflow'
    plan_path, render_path = directory/(stem+'.plan.json'), directory/(stem+'.render.json')
    plan = load(plan_path)
    protocol_path = directory/stem/'api.protocol.json'
    protocol, api = load(protocol_path), load(directory/stem/'api.json')
    require(api['complete'] is True and api['protocol_sha256'] == digest(protocol_path) and
            protocol['plan_sha256'] == digest(plan_path) and protocol['render_sha256'] == digest(render_path) and
            len(api['http_attempts']) == len(api['clips']) == 6, 'Guarded workflow checks incomplete')
    for name, expected in protocol['component_quality_reports_sha256'].items():
        require(digest(directory/name) == expected, 'Workflow qualification evidence changed')
    components = {r['id']: r for component in ['vocos_encodec_development', 'codec_half_donor_dev']
                  for r in followup_quality(directory, component)}
    rows = []
    for item in plan['items']:
        row = dict(components[item['id']])
        q = criteria(row)
        require(all(q.values()) and q == protocol['quality_qualification_before_detector_labels'][row['id']],
                'Workflow selected file did not pass all fixed quality criteria')
        checks = [c for c in api['clips'] if c['id'] == row['id']]
        require(sorted(c['role'] for c in checks) == ['processed_1', 'processed_2'] and
                all(c['http_status'] == 200 and c['sha256'] == row['output_sha256'] and
                    type(c['detected']) is bool for c in checks), 'Workflow checked different bytes')
        row.update(detector_checked=True, estimated_quality_criteria=q,
                   processed_detected=[c['detected'] for c in checks])
        row['status'] = 'still_detected' if any(row['processed_detected']) else 'checked_negative_twice'
        rows.append(row)
    controls = [c for c in api['clips'] if c['role'].endswith('positive_control')]
    require(len(controls) == 2 and all(c['http_status'] == 200 and c['detected'] is True for c in controls),
            'Guarded workflow service controls failed')
    passed = all(not any(r['processed_detected']) for r in rows)
    require(passed == api['all_outputs_double_negative'], 'Workflow summary contradicts responses')
    return {'scope': protocol['scope'], 'source_recordings': 2, 'fresh_recordings': 0,
            'decision_rule': plan['decision_rule'], 'decisions': plan['decisions'],
            'protocol_sha256': digest(protocol_path), 'http_attempts': 6, 'positive_service_controls': 2,
            'cases': rows, 'all_outputs_double_negative': passed,
            'promoted_to_runtime': False}


def rate_controls(directory):
    folder = directory/'source_rate_roundtrip_dev'
    protocol, api = load(folder/'protocol.json'), load(folder/'api.json')
    require(api['complete'] is True and api['protocol_sha256'] == digest(folder/'protocol.json') and
            len(api['http_attempts']) == len(api['clips']) == 6 and
            api['rate_controls_positive_on_both_checks'] is True, 'Incomplete rate controls')
    rows = []
    for item in protocol['items']:
        require(digest(ROOT/item['output']) == item['output_sha256'], 'Roundtrip source changed')
        observations = [c for c in api['clips'] if c['id'] == item['id']]
        require(len(observations) == 2 and all(c['http_status'] == 200 and c['detected'] is True and
                c['sha256'] == item['output_sha256'] for c in observations), 'Roundtrip label mismatch')
        rows.append({k: item[k] for k in ('id', 'source_sha256', 'output_sha256', 'sample_rate', 'samples', 'channels')} |
                    {'detected': [c['detected'] for c in observations]})
    controls = [c for c in api['clips'] if c['role'].endswith('positive_control')]
    require(len(controls) == 2 and all(c['detected'] is True and c['http_status'] == 200 for c in controls),
            'Roundtrip service controls failed')
    return {'scope': protocol['scope'], 'http_attempts': 6, 'positive_service_controls': 2, 'cases': rows}


def build(directory):
    long = load(directory/'mms_whole_long_preflight.json')
    require(long['complete'] is True and long['whole_source_unsegmented'] is True and
            long['candidate_outputs_not_scored'] is True and long['calibration_eligible'] is True,
            'Independent whole-long original review did not qualify')
    encoding = load(directory/'mms_encoding_only_controls.json')
    require(encoding['complete'] is True and len(encoding['clips']) == 2 and
            all(r['raw_canonical_paired_disagreement'] == 0 for r in encoding['clips']), 'Encoding ASR control failed')
    additional = load(directory/'fastconformer_ar_review/result.json')
    calibration = load(directory/'fastconformer_ar_review/original_calibration.json')
    require(additional['complete'] is True and calibration['candidate_outputs_not_read'] is True and
            additional['source_originals'] == calibration['source_originals'], 'Additional review was not original-first')
    codec = checked_component(directory, 'vocos_encodec_development', 4, 6)
    bands = checked_component(directory, 'codec_vocoder_band_dev', 4, 8)
    return {'schema': 2, 'scope': 'Development on two already-spent generated removal sources; independent original-only calibration supplements',
            'removal_development_sources': 2, 'fresh_efficacy_recordings': 0, 'new_tts_requests': 0,
            'status': 'A development selector has quality-qualified twice-negative outputs for both spent sources; runtime and fresh validation are pending.',
            'strict_estimated_quality_thresholds_unchanged': True,
            'seed_resampling': seed_component(directory),
            'vocos_mel': checked_component(directory, 'vocos_development', 2, 4),
            'vocos_encodec': codec,
            'codec_word_repairs': repair_components(directory),
            'fixed_guarded_workflow': guarded_workflow(directory),
            'native_decoder_and_cycles': checked_component(directory, 'codec_decoder_cycle_dev', 10, 6),
            'fullband_dac': quality_rejected_component(directory, 'dac_fullband_dev', 4, 2),
            'partial_strength_repair': quality_rejected_component(directory, 'codec_partial_repair_dev', 2, 1),
            'native_arabic_diagnostic': arabic_diagnostic(directory),
            'codec_vocoder_band_mixtures': bands,
            'standalone_band_probe_parity': band_mix_parity(directory),
            'next_development_selector': development_selector(codec, bands),
            'vocos_mel_assets': load(directory/'vocos_assets.json'),
            'vocos_encodec_assets': load(directory/'vocos_encodec_assets.json'),
            'vocos_model_bandwidth_limit_hz': 12000, 'dac_model_sample_rate': 44100,
            'original_format_and_length_restored': True,
            'sample_rate_roundtrip_controls': rate_controls(directory),
            'encoding_only_asr_controls': encoding, 'whole_long_original_calibration': long,
            'additional_arabic_review': additional,
            'additional_arabic_followup_review': additional_arabic_followups(directory),
            'additional_arabic_original_calibration_sha256': digest(directory/'fastconformer_ar_review/original_calibration.json'),
            'limits': ['Two bandwidths or four seeds applied to one recording are treatments of the same recording.',
                       'The codec uses a 24 kHz model; restoring the original sample rate does not restore frequencies above 12 kHz.',
                       'ASR estimates disagree on Arabic. The additional source-calibrated architecture is reported alongside every earlier failure.',
                       'Models are upstream pretrained assets, not project-trained or republished checkpoints.',
                       'Negative detector responses do not prove physical erasure or unchanged speech; no human listening approval was obtained.',
                       'The next selector was formed after observing these spent development results; its runtime has not been implemented or validated.',
                       'No fresh held-out efficacy batch has been generated for these candidates.'],
            'private_recordings_included': 0, 'transcripts_exported': False, 'speaker_embeddings_exported': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--output', type=Path, default=ROOT/'results/reconstruction_codec_development.json')
    args = parser.parse_args()
    report = build(args.directory)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Exported spent codec/seed evidence, retained Arabic failures and independent calibration controls')


if __name__ == '__main__':
    main()
