from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import run_codec_profiles as study


class FrozenProfileServicePhases(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audio = self.root/'original.flac'; self.audio.write_bytes(b'whole-file fixture')
        authored_item = {'id': 'one', 'model': 'sonic-preview', 'voice': 'fixture voice',
                         'language': 'en', 'text': 'authored reference', 'marked_provenance': True,
                         'verified_output': str(self.root/'out.flac'), 'audit_output': str(self.root/'audit.json')}
        authored = self.root/'authored.json'
        authored.write_text(json.dumps({'items': [authored_item]}))
        self.freeze = self.root/'freeze.json'
        self.freeze.write_text(json.dumps({'source_files_sha256': {}, 'generation_plan_path': str(authored),
                                           'generation_plan_sha256': study.digest(authored),
                                           'maximum_http_attempts_per_phase': {'baseline': 2, 'candidates': 10}}))
        generation = {'complete': True, 'generation_plan_sha256': study.digest(authored),
                      'freeze_sha256': study.digest(self.freeze), 'requests': [
                          dict(authored_item, source=str(self.audio), sha256=study.digest(self.audio))]}
        (self.root/'generation.json').write_text(json.dumps(generation))
        self.plan = self.root/'evaluation.plan.json'
        self.value = {'freeze_sha256': study.digest(self.freeze), 'generation_plan_sha256': study.digest(authored),
                      'generation_receipt_sha256': study.digest(self.root/'generation.json'),
                      'items': [dict(authored_item, source=str(self.audio), sha256=study.digest(self.audio),
                                     output=authored_item['verified_output'])]}
        self.plan.write_text(json.dumps(self.value))

    def run_baseline(self, responses):
        responses = iter(responses)
        def detect(path):
            result = study.cartesia.requests.post('mock service')
            return result.status_code, {'detected': next(responses)}
        with patch.object(study.cartesia.requests, 'post', return_value=SimpleNamespace(status_code=200)) as post, \
                patch.object(study.cartesia, 'detect', side_effect=detect):
            study.endpoint_phase(self.plan, self.freeze, self.root, 'baseline')
            return post.call_count

    def test_baseline_retains_whole_file_and_both_actual_mock_requests(self):
        self.assertEqual(self.run_baseline([True, True]), 2)
        report = json.loads((self.root/'baseline.json').read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(len(report['http_attempts']), 2)
        self.assertEqual(len(report['cases']), 1)
        self.assertTrue(all(c['sha256'] == study.digest(self.audio) for c in report['checks']))

    def test_conflicting_responses_stop_and_existing_observations_cannot_be_repeated(self):
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            self.run_baseline([True, False])
        report = json.loads((self.root/'baseline.json').read_text())
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['http_attempts']), 2)
        with self.assertRaisesRegex(ValueError, 'never silently repeated'):
            self.run_baseline([])

    def test_changed_generation_binding_stops_before_any_service_request(self):
        self.value['items'][0]['sha256'] = 'different'
        self.plan.write_text(json.dumps(self.value))
        with patch.object(study.cartesia, 'detect') as detector:
            with self.assertRaisesRegex(ValueError, 'generation receipts'):
                study.endpoint_phase(self.plan, self.freeze, self.root, 'baseline')
            detector.assert_not_called()

    def test_candidate_phase_requires_all_original_only_calibrations_before_models(self):
        self.run_baseline([True, True])
        with patch.object(study, 'CodecRuntime') as runtime, patch.object(study.cartesia, 'detect') as detector:
            with self.assertRaises(FileNotFoundError):
                study.endpoint_phase(self.plan, self.freeze, self.root, 'candidates')
            runtime.assert_not_called()
            detector.assert_not_called()

    def test_materialized_reference_text_cannot_be_changed_after_authoring(self):
        self.value['items'][0]['text'] = 'a different reference'
        self.plan.write_text(json.dumps(self.value))
        with patch.object(study.cartesia, 'detect') as detector:
            with self.assertRaisesRegex(ValueError, 'authored request'):
                study.endpoint_phase(self.plan, self.freeze, self.root, 'baseline')
            detector.assert_not_called()


if __name__ == '__main__':
    unittest.main()
