"""Export every case using the already frozen profile-scoring arithmetic.

Formatting this report does not select candidates or change the frozen models,
normalization, thresholds, input plan or qualification function.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest
from codec_content import assess
from codec_profiles import method_order
from codec_text import text_units, paired_error
from codec_word_consensus import comparison_reference
from speech_repair import pcm_digest
from experiments.codec_profile_cache import read_recognition
from experiments.codec_profile_evaluation import evaluation_metrics, quality_criteria
from experiments.run_codec_profiles import frozen_sources, validate_materialized


def load(path):
    value = json.loads(path.read_text())
    if value.get('complete') is not True:
        raise ValueError('Required measurement is incomplete: '+path.name)
    return value


def protocol(report, minimum=80):
    checks, attempts = report['checks'], report['http_attempts']
    if len(checks) != len(attempts):
        raise ValueError('Every actual detector attempt needs its corresponding response')
    for check, attempt in zip(checks, attempts):
        if any(check[k] != attempt[k] for k in ('id', 'role', 'ordinal', 'sha256', 'http_status')) or check['http_status'] != 200 or type(check['detected']) is not bool:
            raise ValueError('Detector receipt identity or response changed')
    starts = [datetime.fromisoformat(c['started_at']) for c in attempts]
    if any((b-a).total_seconds() < minimum-.1 for a,b in zip(starts,starts[1:])):
        raise ValueError('The frozen detector pacing changed')
    consistent_replies(checks)


def consistent_replies(checks):
    seen = {}
    for check in checks:
        if type(check.get('detected')) is not bool:
            continue
        identity = (check['id'], check['role'].removesuffix('_positive_confirmation'), check['sha256'])
        if identity in seen and seen[identity] != check['detected']:
            raise ValueError('Repeated whole-file detector responses conflict')
        seen[identity] = check['detected']


def index_rows(report, expected, key):
    rows = {key(r): r for r in report['clips']}
    if len(rows) != len(report['clips']) or set(rows) != expected:
        raise ValueError('Keep every planned case and declared checkpoint exactly once')
    return rows


def original_primary_sha256(freeze):
    path = 'results/codec_reliability.json'
    committed = subprocess.run(['git','show',freeze['candidate_implementation_commit']+':'+path],
                               cwd=ROOT,check=True,capture_output=True).stdout
    checksum = hashlib.sha256(committed).hexdigest()
    if digest(ROOT/path) != checksum:
        raise ValueError('The original primary study differs from its frozen historical bytes')
    return checksum


def export(freeze_path, directory):
    import numpy as np
    import soundfile as sf
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    freeze = json.loads(freeze_path.read_text())
    frozen_sources(freeze)
    plan_path = directory/'evaluation.plan.json'
    plan = json.loads(plan_path.read_text())
    validate_materialized(plan, freeze, freeze_path, directory)
    plan_sha = digest(plan_path)
    generation = load(directory/'generation.json')
    baseline, candidates = load(directory/'baseline.json'), load(directory/'candidates.json')
    for report in (baseline, candidates):
        if report['plan_sha256'] != plan_sha or report['freeze_sha256'] != digest(freeze_path):
            raise ValueError('Endpoint observations belong to a different frozen study')
        protocol(report, freeze['minimum_detector_start_interval_seconds'])
        if len(report['http_attempts']) > freeze['maximum_http_attempts_per_phase'][report['phase']]:
            raise ValueError('A frozen request ceiling was exceeded')
    ids = {i['id'] for i in plan['items']}
    if len(plan['items']) != freeze['planned_requests']:
        raise ValueError('The planned denominator changed')
    originals = {c['id']:c for c in baseline['cases']}
    selected = {c['id']:c for c in candidates['cases']}
    if (set(originals) != ids or set(selected) != ids or len(originals) != len(baseline['cases']) or
            len(selected) != len(candidates['cases'])):
        raise ValueError('Every baseline and selected/not-cleared case must remain')
    positive = [i for i in plan['items'] if i['marked_provenance'] and originals[i['id']]['detected']]
    if not positive:
        raise ValueError('Measured-positive marked controls are required')
    for records,item,role in ((candidates['checks'][:2],positive[0],'opening_positive_control'),
                              (candidates['checks'][-2:],positive[-1],'closing_positive_control')):
        if (len(records)!=2 or [c['ordinal'] for c in records]!=[1,2] or
                any(c['id']!=item['id'] or c['sha256']!=item['sha256'] or c['role']!=role or c['detected'] is not True for c in records)):
            raise ValueError('Keep both opening and closing positive-source controls')
    og = load(directory/'original_guards.json')
    oe, pe = load(directory/'original_evaluation.json'), load(directory/'processed_evaluation.json')
    ov, pv = load(directory/'original_voice.json'), load(directory/'processed_voice.json')
    for report in (og,oe,pe,ov,pv):
        if report['plan_sha256'] != plan_sha:
            raise ValueError('Quality evidence refers to different materialized inputs')
    for name in ('original_guards.json','original_evaluation.json','original_voice.json','baseline.json'):
        if digest(directory/name) != candidates['required_preflight_sha256'][name]:
            raise ValueError('Original-only preflight changed after selection began')
    if not all(r['originals_only'] and not r['candidate_outputs_read'] for r in (og,oe,ov)):
        raise ValueError('Original calibration must precede candidate grading')
    if oe['checkpoints'] != pe['checkpoints']:
        raise ValueError('Independent evaluator metadata changed between original and processed scoring')
    original_guards = index_rows(og, ids, lambda r:r['id'])
    metrics = {(i['id'],m) for i in plan['items'] for m in evaluation_metrics(i['language'])}
    original_eval, processed_eval = index_rows(oe,metrics,lambda r:(r['id'],r['metric'])), index_rows(pe,metrics,lambda r:(r['id'],r['metric']))
    original_voice, processed_voice = index_rows(ov,ids,lambda r:r['id']), index_rows(pv,ids,lambda r:r['id'])
    normalizer = BasicTextNormalizer(remove_diacritics=False)
    results = []
    for item in plan['items']:
        case, source = item['id'], ROOT/item['source']
        source_sha = digest(source)
        if source_sha != item['sha256'] or originals[case]['source_sha256'] != source_sha:
            raise ValueError('Canonical original changed')
        checks = [c for c in baseline['checks'] if c['id']==case]
        if (len(checks)!=2 or [c['ordinal'] for c in checks]!=[1,2] or
                any(c['role']!='original' or c['sha256']!=source_sha or c['detected']!=originals[case]['detected'] for c in checks)):
            raise ValueError('Every source needs two consistent whole-file baseline observations')
        info = sf.info(source)
        control = original_guards[case]
        cached = directory/'recognition/originals'/(source_sha+'.json')
        if digest(cached) != control['recognition_cache_sha256']:
            raise ValueError('Original recognition changed after calibration')
        before = read_recognition(directory/'recognition/originals',source_sha,og['configuration_sha256'])
        record = selected[case]
        orders = method_order(before['language'],info.frames/info.samplerate) if before['supported'] else ('codec-native',)
        if tuple(record['method_order']) != orders or [a['preset'] for a in record['attempts']] != list(orders[:len(record['attempts'])]):
            raise ValueError('A case did not follow its fixed detected-language/duration order')
        row = {'id':case,'language':item['language'],'detected_language':before['language'],
               'model':item['model'],'voice':item['voice'],'voice_primary_language':item['voice_primary_language'],
               'marked_provenance':item['marked_provenance'],'original_detected':originals[case]['detected'],
               'source_sha256':source_sha,'source_format':{'samples':info.frames,'sample_rate':info.samplerate,'channels':info.channels},
               'selection_status':record['status'],'method_order':list(orders),
               'original_guard_calibration':control,
               'original_independent_calibration':[original_eval[(case,m)] for m in evaluation_metrics(item['language'])],
               'attempts':[{k:a[k] for k in ('preset','state','sha256','detected','runtime_quality_guard','previously_checked_preset') if k in a}
                           for a in record['attempts']],
               'quality_qualified':False,'qualified_marked_transition':False}
        for attempt in record['attempts']:
            if attempt.get('detected') is True:
                evidence = [c for c in candidates['checks'] if c['id']==case and c['role'] in
                            ('processed_'+attempt['preset'],'processed_'+attempt['preset']+'_positive_confirmation')]
                if len(evidence)!=2 or [c['ordinal'] for c in evidence]!=[1,2] or any(
                        c['sha256']!=attempt['sha256'] or c['detected'] is not True for c in evidence):
                    raise ValueError('Every positive candidate counterexample needs its same-file repeat')
        output = ROOT/item['output']
        if record['status']=='verified_negative':
            output_sha = digest(output)
            if output_sha != record['output_sha256']:
                raise ValueError('Selected output bytes changed')
            actual = sf.info(output)
            if (actual.frames,actual.samplerate,actual.channels)!=(info.frames,info.samplerate,info.channels):
                raise ValueError('Selected output changed the complete source grid')
            attempt = record['attempts'][-1]
            metadata = attempt['model']
            audio,_ = sf.read(output,dtype='float32')
            if (not np.array_equal(audio*2**23,np.rint(audio*2**23)) or np.abs(audio).max()>=1 or
                    metadata['content_guard_pcm_sha256']!=pcm_digest(audio) or metadata['content_guard_sample_grid']!='PCM_24'):
                raise ValueError('Selected waveform differs from its assessed PCM24 grid')
            after = read_recognition(directory/'recognition/processed',output_sha,og['configuration_sha256'])
            raw = assess(before,after)
            assessed = raw
            if record['selected_preset']=='codec-band-inverse6-consensus' and before['language']=='ar' and not raw['content_guard_passed']:
                corrected,proof = comparison_reference(before['turbo'],after['turbo'],before['fastconformer'],after['fastconformer'])
                assessed = assess(dict(before,turbo=corrected),after)
                if metadata['raw_content_assessment']!=raw or metadata['consensus_supported_corrections']!=proof:
                    raise ValueError('Consensus changed its raw failure or supported-correction evidence')
            if not assessed['content_guard_passed'] or assessed['paired_disagreement']!=metadata['paired_disagreement']:
                raise ValueError('Final processing guard evidence disagrees with its exact-file cache')
            primary,secondary = assessed['primary_guard'],assessed['secondary_guard']
            truth = text_units(item['text'],normalizer,item['language'])
            source_errors = dict(control['original_reference_error'])
            guard_deltas = {name:None for name in (primary,secondary)}
            if control['language_matches_request']:
                for name in (primary,secondary):
                    error = paired_error(truth,before[name],item['language'])
                    if error!=source_errors[name]:
                        raise ValueError('Original guard reference scoring changed')
                    processed = paired_error(truth,after[name],item['language'])
                    guard_deltas[name] = None if processed is None or error is None else processed-error
            eval_deltas, eval_rows = {}, []
            for metric in evaluation_metrics(item['language']):
                original,measured = original_eval[(case,metric)],processed_eval[(case,metric)]
                if (original['source_sha256']!=source_sha or measured['source_sha256']!=source_sha or
                        measured.get('output_sha256')!=output_sha or measured['original']!=original['original'] or measured['status']!='scored'):
                    raise ValueError('Independent reference evidence refers to other files or calibration')
                before_error,after_error = original['original']['reference_error'],measured['processed']['reference_error']
                delta = None if before_error is None or after_error is None else after_error-before_error
                if delta!=measured['reference_error_delta']:
                    raise ValueError('Independent reference delta changed')
                source_errors[metric]=before_error;eval_deltas[metric]=delta;eval_rows.append(measured)
            voice = processed_voice[case]
            if (voice['source_sha256']!=source_sha or voice.get('output_sha256')!=output_sha or voice['status']!='scored' or
                    voice['original_dnsmos']!=original_voice[case]['original_dnsmos'] or pv['metric']!=ov['metric']):
                raise ValueError('Voice estimates refer to other whole files or estimator versions')
            decision = attempt['runtime_quality_guard']
            if (not decision['passed'] or decision['source_sha256']!=source_sha or decision['candidate_sha256']!=output_sha or
                    decision['speaker_minimum']!=voice['speaker_minimum'] or decision['dnsmos_overall_delta']!=voice['dnsmos_delta']['overall']):
                raise ValueError('Operational voice gates differ from the selected-file estimates')
            negative = [c for c in candidates['checks'] if c['id']==case and c['role']=='processed_'+record['selected_preset']]
            if len(negative)!=2 or [c['ordinal'] for c in negative]!=[1,2] or any(c['sha256']!=output_sha or c['detected'] is not False for c in negative):
                raise ValueError('Retain two actual negatives for the exact selected whole file')
            qualification = quality_criteria(item['language'],assessed['paired_disagreement'],guard_deltas,eval_deltas,source_errors,
                voice['speaker_minimum'],voice['dnsmos_delta']['overall'],assessed['content_guard_passed'],control['language_matches_request'])
            row.update(qualification,output_sha256=output_sha,selected_preset=record['selected_preset'],raw_content_assessment=raw,
                       final_content_assessment=assessed,original_reference_error=source_errors,guard_reference_delta=guard_deltas,
                       independent_evaluation=eval_rows,voice=voice,negative_checks=negative)
            row['qualified_marked_transition']=item['marked_provenance'] and originals[case]['detected'] and qualification['quality_qualified']
        elif output.exists():
            raise ValueError('An unselected case unexpectedly has a published output')
        results.append(row)
    marked = [r for r in results if r['marked_provenance']]
    controls = [r for r in results if not r['marked_provenance']]
    primary_sha = original_primary_sha256(freeze)
    return {'kind':'Frozen prospective language-profile study','freeze_sha256':digest(freeze_path),
            'generation_plan_sha256':freeze['generation_plan_sha256'],'candidate_implementation_commit':freeze['candidate_implementation_commit'],
            'planned_requests':len(results),'marked_provenance_requests':len(marked),'distinct_marked_voice_ids':len({r['voice'] for r in marked}),
            'matched_nominal_controls':len(controls),'marked_originals_detected':sum(r['original_detected'] for r in marked),
            'qualified_marked_transitions':sum(r['qualified_marked_transition'] for r in marked),
            'quality_qualified_control_outputs':sum(r['quality_qualified'] for r in controls),'cases':results,
            'http_attempt_counts':{r['phase']:len(r['http_attempts']) for r in (baseline,candidates)},
            'endpoint_receipts':{'baseline':baseline,'candidates':{'checks':candidates['checks'],'http_attempts':candidates['http_attempts']}},
            'budget_at_freeze':freeze['budget'],'original_primary_study_sha256':primary_sha,'original_primary_study_result_changed':False,
            'human_listening_approval':False,'physical_erasure_proven':False,
            'limits':['All planned requests remain; nominal controls, repeated calls and fallback treatments are not extra removal successes.',
                      'Processing guards and operational voice estimates guide selection; only the declared evaluation-ASR reference scores remain withheld.',
                      'Whisper checkpoints share ancestry; speaker/MOS estimates are uncalibrated proxies without human listening approval.',
                      'Some public primary-language tags differ from requested utterance languages; both are retained.',
                      'A new recording and voice ID do not establish unseen training data for upstream models.',
                      'Whole-file detector negatives establish current checked behavior, not physical erasure or universal reliability.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=export(args.freeze,args.directory)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PROFILE_STUDY_EXPORTED',report['qualified_marked_transitions'],'/',report['marked_provenance_requests'])
