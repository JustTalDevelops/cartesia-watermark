"""Curate spent-file precision experiments without exporting recognition text."""
import argparse
import json
from math import isfinite
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from codec_text import paired_error
from experiments.codec_inference_followup_report import verify_source
from experiments.codec_profiles_report import original_primary_sha256
from experiments.run_codec_profiles import frozen_sources
from experiments.vocoder_codec_tradeoff_report import digest,load


def input_files(directory):
    plan=json.loads((directory/'codec_fresh/evaluation.plan.json').read_text())
    originals={i['language']+('_2' if 'ta2' in i['id'] else ''):i for i in plan['items']
               if i['marked_provenance'] and i['language'] in ('hi','ta')}
    paths={key+'_original':ROOT/i['source'] for key,i in originals.items()}
    paths.update(hi_selected=ROOT/originals['hi']['output'],
                 ta_band_prior=directory/'lazy_codec_integrations/ta1/candidate.flac',
                 ta2_band_prior=directory/'profile_prior_reuse/0.flac',
                 ta2_narrow=directory/'profile_prior_reuse/1.flac',
                 ta2_mimi_rejected=directory/'mimi_ta2/32/candidate.flac')
    return paths


def component(directory,precision,freeze):
    folder=directory/('large_dynamic_'+precision)
    report=load(folder/'result.json')
    verify_source({'implementation_commit':report['frozen_implementation_commit'],
                   'source_files_sha256':report['frozen_source_files_sha256']})
    if (report['frozen_source_files_sha256']!=freeze['source_files_sha256'] or
            report['fresh_independent_sources']!=0 or report['new_http_requests']!=0 or report['new_tts_requests']!=0 or
            report['reference_text_used'] or report['threads']!=4 or report['quantized_linear_modules']!=512 or
            digest(directory/('probe_large_dynamic_'+precision+'.py'))!=report['benchmark_source_sha256']):
        raise ValueError('Precision benchmark changed its source, assets or spent-only scope')
    private=json.loads((folder/'private_units.json').read_text())
    before,after=private['float32'],private[precision]
    rows={r['id']:r for r in report['cases']};paths=input_files(directory)
    if len(rows)!=8 or len(rows)!=len(report['cases']) or set(rows)!=set(paths) or set(before)!=set(rows) or set(after)!=set(rows):
        raise ValueError('Retain every original and accepted/rejected component fixture')
    checks=[]
    for case,row in rows.items():
        old,new=before[case],after[case]
        if (digest(paths[case])!=row['audio_sha256'] or row['language'] not in ('hi','ta') or
                row['normalized_units_identical']!=(old['units']==new['units']) or
                row['token_ids_identical']!=(old['token_ids']==new['token_ids']) or
                row['precision_disagreement']!=paired_error(old['units'],new['units'],row['language']) or
                row['float32_seconds']!=old['generation_seconds'] or row[precision+'_seconds']!=new['generation_seconds'] or
                not all(isfinite(t) and t>0 for t in (row['float32_seconds'],row[precision+'_seconds']))):
            raise ValueError('Precision timings or transcription comparisons do not bind their measured files')
        if row['source_id']!=case:
            a=paired_error(before[row['source_id']]['units'],old['units'],row['language'])
            b=paired_error(after[row['source_id']]['units'],new['units'],row['language'])
            checks.append({'id':case,'float32_paired_disagreement':a,precision+'_paired_disagreement':b,
                           'single_primary_gate_unchanged':(a is not None and a<=.01+1e-12)==(b is not None and b<=.01+1e-12)})
    if checks!=report['paired_checks'] or len(checks)!=5:
        raise ValueError('Every original-to-candidate primary comparison must remain')
    rejected=next(r for r in checks if r['id']=='ta2_mimi_rejected')
    if rejected['float32_paired_disagreement']<=.01:
        raise ValueError('The rejected fixture lost its original primary rejection')
    old_total=sum(r['float32_seconds'] for r in rows.values());new_total=sum(r[precision+'_seconds'] for r in rows.values())
    report.update(summary={'spent_source_recordings':3,'audio_inputs':8,'normalized_transcript_matches':sum(r['normalized_units_identical'] for r in rows.values()),
                           'exact_token_matches':sum(r['token_ids_identical'] for r in rows.values()),
                           'primary_decision_changes':[r['id'] for r in checks if not r['single_primary_gate_unchanged']],
                           'float32_generation_seconds':old_total,precision+'_generation_seconds':new_total,
                           'ratio_of_total_generation_times':old_total/new_total},
                  private_observations_sha256=digest(folder/'private_units.json'))
    return report,private


def workflow(directory):
    baseline=load(directory/'profile_cli_integration_corrected/result.json')
    verify_source(baseline)
    folder=directory/'profile_cli_integration_packed_fp16'
    packed=load(folder/'result.json');verify_source(packed)
    audit=json.loads((folder/'audit.json').read_text())
    if (packed['fresh_independent_sources']!=0 or packed['new_http_requests']!=0 or packed['new_tts_requests']!=0 or
            packed['reference_text_used'] or packed['detector_transport']!='offline injected mock' or packed['original_recognition_inferences']!=1 or
            digest(folder/'candidate.flac')!=baseline['canonical_output_sha256'] or
            digest(directory/'profile_cli_integration_corrected/candidate.flac')!=baseline['canonical_output_sha256'] or
            digest(directory/'profile_cli_integration_corrected/audit.json')!=baseline['audit_sha256'] or
            digest(folder/'audit.json')!=packed['audit_sha256'] or packed['mocked_detector_calls']!=baseline['mocked_detector_calls'] or
            packed['source_sha256']!=baseline['source_sha256'] or packed['canonical_output_sha256']!=baseline['canonical_output_sha256'] or
            not packed['temporary_recognition_and_voice_caches_removed']):
        raise ValueError('Packed wrapper lost its exact-file, mocked-only integration scope')
    if ([(c['sha256'],c['detected']) for c in audit['checks']]!=
            [(c['sha256'],c['mocked_detected']) for c in packed['mocked_detector_calls']] or
            len(audit['checks'])!=8 or [c['source_job_reused'] for c in packed['candidates']]!=[False,True] or
            [c['band_prior_reused'] for c in packed['candidates']]!=[False,True]):
        raise ValueError('Keep every mocked check and the measured source/prior reuse')
    packing=audit['processing_recognizers']['large']['weight_packing']
    if packing['implementation_sha256']!=packed['source_files_sha256']['tools/codec_packed_fp16.py'] or packing['weights']!='torch.float16':
        raise ValueError('Wrapper did not declare the measured packed recognizer')
    for a,b in zip(baseline['candidates'],packed['candidates'],strict=True):
        if (a['canonical_output_sha256']!=b['canonical_output_sha256'] or a['paired_disagreement']!=b['paired_disagreement'] or
                not b['content_guard_passed'] or not b['runtime_voice_guard']['passed'] or
                a['runtime_voice_guard']['speaker_minimum']!=b['runtime_voice_guard']['speaker_minimum'] or
                a['runtime_voice_guard']['dnsmos_overall_delta']!=b['runtime_voice_guard']['dnsmos_overall_delta']):
            raise ValueError('Packed wrapper changed a file, speech comparison or voice estimate')
    packed['comparison']={'float32_wrapper_seconds':baseline['elapsed_seconds'],'packed_wrapper_seconds':packed['elapsed_seconds'],
                          'ratio_of_total_times':baseline['elapsed_seconds']/packed['elapsed_seconds'],
                          'scope':'One complete spent-source local run per configuration, including construction/calibration; all detector replies explicitly mocked'}
    return packed


def export(directory,freeze_path):
    freeze=json.loads(freeze_path.read_text());frozen_sources(freeze)
    int8,raw8=component(directory,'int8',freeze);fp16,raw16=component(directory,'fp16',freeze)
    if (raw8['float32']!=raw16['float32'] or int8['model']!=fp16['model'] or
            fp16['float32_reference_report_sha256']!=digest(directory/'large_dynamic_int8/result.json') or
            fp16['float32_reference_units_sha256']!=digest(directory/'large_dynamic_int8/private_units.json') or
            not fp16['float32_sanity_units_identical'] or not fp16['float32_sanity_tokens_identical'] or
            fp16['observed_linear_output_dtype']!='torch.float32' or fp16['packed_weight_dtype']!='torch.float16'):
        raise ValueError('FP16 comparison lost its actual earlier FP32 reference or sanity reproduction')
    return {'kind':'Spent-file Large-V3 precision development','fresh_independent_sources':0,'new_tts_requests':0,'new_http_requests':0,
            'new_removal_successes':0,'hardware':'AMD Ryzen 9 9900X; CPU quota four cores; no GPU',
            'int8_component':int8,'packed_fp16_component':fp16,'packed_fp16_workflow':workflow(directory),
            'active_frozen_study_uses_packed_weights':False,'original_primary_study_sha256':original_primary_sha256(freeze),
            'human_listening_approval':False,'physical_erasure_proven':False,
            'limits':['Eight audio files are variants of three spent original recordings; no fresh independent sources or endpoint observations.',
                      'Component measurements use forced known Hindi/Tamil language and one warmed full generation per file; loading/features/packing are excluded.',
                      'FP16 reuses the bound earlier FP32 measurements and separately reproduces its first-source reference; these are individual runs without statistical speed guarantees.',
                      'INT8 changed a primary guard decision and is not integrated. Packed FP16 remains experimental and outside the active frozen study.',
                      'Matching these fixtures does not prove unchanged recognition on arbitrary recordings or reliable removal.',
                      'The whole-wrapper comparison uses real speech/voice models and explicitly mocked detector replies, which are never counted as removal evidence.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'.cache/removal-reliability')
    parser.add_argument('--freeze',type=Path,default=ROOT/'results/codec_profiles_reliability.freeze.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/codec_precision_development.json')
    args=parser.parse_args();report=export(args.directory,args.freeze)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output)
