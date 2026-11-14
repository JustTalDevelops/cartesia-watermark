from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments.codec_profiles_report import protocol,index_rows,summarize_attempt


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


if __name__=='__main__':unittest.main()
