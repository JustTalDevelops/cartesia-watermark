"""Offline checks of request accounting; no remote detector or neural models."""
from collections import deque
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests
import soundfile as sf

from experiments import check_codec_frames as checker


class FrameEndpointTests(unittest.TestCase):
    def fixture(self, folder, historical_candidate=None):
        root = Path(folder)
        files = []
        for number in range(3):
            target = root/f'{number}.flac'
            t = np.arange(40000, dtype=np.float32)/8000
            sf.write(target, .1*np.sin(2*np.pi*(231+number)*t), 8000, subtype='PCM_24')
            files.append({'id':f'case{number}', 'file':str(target), 'sha256':checker.digest(target)})
        generation, study = root/'generation.json', root/'study.json'
        generation.write_text('{}'); study.write_text('{}')
        candidate = dict(files[2], id='case0', offset_native_samples=80,
                         detector_eligible=True, processing_status='processing_eligible')
        duplicate = dict(candidate, offset_native_samples=160)
        rejected = {'id':'case0', 'offset_native_samples':0, 'detector_eligible':False,
                    'processing_status':'content_rejected', 'sha256':'not-checked'}
        historical = {r['sha256']:True for r in files[:2]}
        if historical_candidate is not None:
            historical[candidate['sha256']] = historical_candidate
        value = {'generation_file':str(generation), 'generation_sha256':checker.digest(generation),
                 'completed_study_file':str(study), 'completed_study_sha256':checker.digest(study),
                 'code_files_sha256':{}, 'sources':files[:2], 'treatments':[rejected,candidate,duplicate],
                 'maximum_http_attempts':6, 'historical_exact_file_observations':historical}
        declaration = root/'declaration.json'
        declaration.write_text(json.dumps(dict(value, declared_at=checker.now())))
        return value, declaration, root/'result.json'

    def execute(self, value, declaration, output, replies, pace=None):
        responses = deque(replies)
        def response(*args, **kwargs):
            value = responses.popleft()
            if isinstance(value, Exception): raise value
            status, detected = value if isinstance(value, tuple) else (200, value)
            return Mock(status_code=status, json=lambda:{'detected':detected})
        post = Mock(side_effect=response)
        with patch.object(checker, 'declaration_data', return_value=value), \
                patch.object(checker.cartesia.requests, 'post', post), \
                patch.object(checker.cartesia, 'ROOT', declaration.parent), \
                patch.object(checker.cartesia, '_pace_detector', side_effect=pace or (lambda:None)):
            checker.run(declaration, output)
        return post

    def test_fixed_order_two_actual_checks_per_file_and_duplicate_rows_remain(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            post = self.execute(value,declaration,output,[True,True,False,False,True,True])
            report = json.loads(output.read_text())
            self.assertTrue(report['complete']);self.assertEqual(post.call_count,6)
            self.assertEqual(len(report['checks']),len(report['http_attempts']))
            self.assertEqual([r['role'] for r in report['checks']],
                ['opening_positive_control']*2+['candidate']*2+['closing_positive_control']*2)
            self.assertEqual(len(report['treatments']),3)
            self.assertFalse(report['treatments'][1]['identical_file_observations_reused'])
            self.assertTrue(report['treatments'][2]['identical_file_observations_reused'])
            self.assertEqual(report['treatments'][1]['detector_observation_group'],
                             report['treatments'][2]['detector_observation_group'])

    def test_connection_failure_is_one_uncertain_attempt_without_client_retry(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            with self.assertRaisesRegex(RuntimeError,'no automatic retry'):
                self.execute(value,declaration,output,[requests.ConnectionError('offline fixture')])
            report = json.loads(output.read_text())
            self.assertFalse(report['complete']);self.assertEqual(len(report['http_attempts']),1)
            self.assertEqual(report['checks'],[])

    def test_error_response_is_preserved_without_a_repeat(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                self.execute(value,declaration,output,[(503,None)])
            report = json.loads(output.read_text())
            self.assertEqual([a['http_status'] for a in report['http_attempts']],[503])

    def test_inconsistent_candidate_repeat_stops_before_later_treatments(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            with self.assertRaisesRegex(RuntimeError,'Contradictory'):
                self.execute(value,declaration,output,[True,True,False,True])
            report = json.loads(output.read_text())
            self.assertFalse(report['complete']);self.assertEqual(len(report['http_attempts']),4)
            self.assertEqual(report['observations'],{})

    def test_new_observation_cannot_override_a_known_same_file_result(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder, historical_candidate=True)
            with self.assertRaisesRegex(RuntimeError,'Contradictory'):
                self.execute(value,declaration,output,[True,True,False])
            self.assertEqual(len(json.loads(output.read_text())['http_attempts']),3)

    def test_changed_file_during_pacing_never_reaches_transport(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            calls = []
            def pace():
                calls.append(True)
                if len(calls)==3:
                    Path(value['treatments'][1]['file']).write_bytes(b'changed while queued')
            with self.assertRaisesRegex(ValueError,'exact detector input changed'):
                self.execute(value,declaration,output,[True,True],pace=pace)
            self.assertEqual(len(json.loads(output.read_text())['http_attempts']),2)

    def test_existing_observations_and_changed_declarations_are_not_replayed(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            output.write_text('{}')
            with self.assertRaisesRegex(ValueError,'never automatically repeated'):
                self.execute(value,declaration,output,[])
            output.unlink()
            bad = json.loads(declaration.read_text());bad['maximum_http_attempts']=99
            declaration.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'inventory'):
                self.execute(value,declaration,output,[])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
