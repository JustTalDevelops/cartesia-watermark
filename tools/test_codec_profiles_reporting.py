from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments.codec_profiles_report import protocol,index_rows,summarize_attempt,attempt_speech_evidence
from codec_content import assess
from codec_word_consensus import comparison_reference


class ProfileEvidenceValidation(unittest.TestCase):
    def setUp(self):
        start=datetime(2026,11,14,tzinfo=timezone.utc)
        self.report={'checks':[{'id':'one','role':'original','ordinal':i+1,'sha256':'whole','http_status':200,'detected':True} for i in range(2)],
                     'http_attempts':[{'id':'one','role':'original','ordinal':i+1,'sha256':'whole','http_status':200,
                                       'started_at':(start+timedelta(seconds=80*i)).isoformat()} for i in range(2)]}

    def test_complete_matching_receipts_are_accepted(self):
        protocol(self.report)

    def test_unmatched_request_changed_uploaded_bytes_and_missing_pacing_are_rejected(self):
        for mutation in ('missing','changed','pacing'):
            report=deepcopy(self.report)
            if mutation=='missing':report['checks'].pop()
            elif mutation=='changed':report['checks'][1]['sha256']='other file'
            else:report['http_attempts'][1]['started_at']=report['http_attempts'][0]['started_at']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):protocol(report)

    def test_missing_or_duplicate_cases_cannot_shrink_the_denominator(self):
        for rows in ([{'id':'one'}],[{'id':'one'},{'id':'one'}]):
            with self.assertRaises(ValueError):index_rows({'clips':rows},{'one','two'},lambda r:r['id'])

    def test_negative_then_positive_is_inconsistent_even_if_positive_repeats(self):
        report=deepcopy(self.report)
        report['checks'][0]['detected']=False
        with self.assertRaisesRegex(ValueError,'conflict'):protocol(report)

    def test_rejection_keeps_unmeasured_primary_null_and_omits_raw_recognition(self):
        attempt={'preset':'codec-band','state':'content_rejected','model':{
            'content_guard_passed':False,'paired_disagreement':{'large':None,'mms':.12},
            'maximum_paired_disagreement':.01,'initial_primary_skip_reason':'secondary_recognizer_rejected_candidate',
            'rejection_reason':'recognizer_disagreement','transcript':'PRIVATE_TRANSCRIPT',
            'alignment':{'words':['PRIVATE_TRANSCRIPT']},'embeddings':[.2,.3]}}
        summary=summarize_attempt(attempt)
        self.assertIsNone(summary['speech_assessment']['paired_disagreement']['large'])
        self.assertEqual(summary['speech_assessment']['paired_disagreement']['mms'],.12)
        self.assertEqual(summary['speech_assessment']['initial_primary_skip_reason'],
                         'secondary_recognizer_rejected_candidate')
        self.assertNotIn('PRIVATE_TRANSCRIPT',json.dumps(summary))
        self.assertNotIn('embeddings',json.dumps(summary))

    def cached_attempt(self, directory, before, after, state='checked', preset='codec-native'):
        checksum='a'*64
        folder=directory/'recognition/processed'
        folder.mkdir(parents=True)
        path=folder/(checksum+'.json')
        path.write_text(json.dumps({'audio_sha256':checksum,'configuration_sha256':'frozen',
                                    'recognition':after}))
        model=dict(assess(before,after),candidate_flac_sha256=checksum,
                   output_sha256=None if state=='content_rejected' else 'float-container')
        attempt={'preset':preset,'state':state,'model':model}
        if state!='content_rejected':attempt['sha256']=checksum
        return attempt,path

    def test_rejected_cache_is_verified_without_exporting_units_or_imputing_primary(self):
        before={'language':'hi','primary':'large','secondary':'mms',
                'large':['PRIVATE_UNIT'],'mms':['one','two']}
        after=dict(before,large=[],mms=['one','other'])
        with TemporaryDirectory() as temp:
            directory=Path(temp)
            attempt,_=self.cached_attempt(directory,before,after,state='content_rejected')
            evidence=attempt_speech_evidence(before,attempt,directory,'frozen')
            self.assertTrue(evidence['assessment_recomputed'])
            self.assertEqual(len(evidence['recognition_cache_sha256']),64)
            self.assertNotIn('PRIVATE_UNIT',json.dumps(evidence))
            self.assertIsNone(attempt['model']['paired_disagreement']['large'])

    def test_changed_scores_file_binding_configuration_and_rejection_state_are_rejected(self):
        before={'language':'fr','primary':'turbo','secondary':'mms',
                'turbo':['one'],'mms':['one']}
        for mutation in ('score','sha256','configuration','state','unit_count'):
            with self.subTest(mutation=mutation),TemporaryDirectory() as temp:
                directory=Path(temp)
                attempt,path=self.cached_attempt(directory,before,before)
                if mutation=='score':attempt['model']['paired_disagreement']['mms']=.1
                elif mutation=='sha256':attempt['sha256']='different-file'
                elif mutation=='state':attempt['state']='content_rejected'
                elif mutation=='unit_count':attempt['model']['processed_units']['turbo']=2
                else:
                    value=json.loads(path.read_text());value['configuration_sha256']='other-model'
                    path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):attempt_speech_evidence(before,attempt,directory,'frozen')

    def test_consensus_retains_actual_raw_failure_and_only_bound_corrections(self):
        units=['unit'+str(i) for i in range(40)]
        changed=list(units);changed[10]='first-change';changed[25]='second-change'
        anchor=list(units);anchor[10]=changed[10]
        before={'language':'ar','primary':'turbo','secondary':'fastconformer',
                'turbo':units,'fastconformer':anchor}
        after=dict(before,turbo=changed)
        raw=assess(before,after)
        corrected,proof=comparison_reference(units,changed,anchor,anchor)
        final=assess(dict(before,turbo=corrected),after)
        self.assertFalse(raw['content_guard_passed'])
        self.assertTrue(final['content_guard_passed'])
        with TemporaryDirectory() as temp:
            directory=Path(temp)
            attempt,_=self.cached_attempt(directory,before,after,preset='codec-band-inverse6-consensus')
            attempt['model'].update(final,raw_content_assessment=raw,
                consensus_supported_corrections=proof,consensus_profile='bounded-arabic-word-consensus-v1',
                consensus_reference_text_used=False)
            attempt_speech_evidence(before,attempt,directory,'frozen')
            for mutation in ('raw','proof','reference'):
                damaged=deepcopy(attempt)
                if mutation=='raw':damaged['model']['raw_content_assessment']=final
                elif mutation=='proof':damaged['model']['consensus_supported_corrections']=[]
                else:damaged['model']['consensus_reference_text_used']=True
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    attempt_speech_evidence(before,damaged,directory,'frozen')


if __name__=='__main__':unittest.main()
