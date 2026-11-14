import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import run_codec_study as study


class CodecStudyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.source = self.directory/'original.flac'
        self.source.write_bytes(b'fixture, never uploaded')
        self.checksum = study.digest(self.source)
        self.plan_path, self.freeze_path = self.directory/'plan.json', self.directory/'freeze.json'
        self.item = {'id': 'fixture', 'canonical_output': str(self.source), 'marked_provenance': True,
                     'verified_output': str(self.directory/'output.flac'), 'audit_output': str(self.directory/'audit.json'),
                     'language': 'en', 'text': 'reference withheld from processing'}
        self.plan_path.write_text(json.dumps({'items': [self.item], 'candidate_presets': ['codec-native', 'codec-band']}))
        self.freeze_path.write_text(json.dumps({'generation_plan_sha256': study.digest(self.plan_path),
            'source_files_sha256': {}, 'maximum_http_attempts_per_phase': {'baseline': 2, 'candidates': 8}}))
        (self.directory/'generation.json').write_text(json.dumps({'complete': True,
            'generation_plan_sha256': study.digest(self.plan_path), 'freeze_sha256': study.digest(self.freeze_path),
            'requests': [{'id': 'fixture', 'source': str(self.source), 'sha256': self.checksum}]}))

    def tearDown(self):
        self.temporary.cleanup()

    def test_baseline_repeats_same_whole_file_with_explicit_ordinals(self):
        with patch.object(study.cartesia, 'detect', return_value=(200, {'detected': True})) as detector:
            self.assertEqual(study.run(self.plan_path, self.freeze_path, self.directory, 'baseline'), 0)
        self.assertEqual([c.args[0] for c in detector.call_args_list], [self.source, self.source])
        report = study.load(self.directory/'baseline.json')
        self.assertTrue(report['complete'])
        self.assertEqual([c['ordinal'] for c in report['checks']], [1, 2])
        self.assertEqual([c['sha256'] for c in report['checks']], [self.checksum]*2)
        self.assertEqual(len(report['cases']), 1)

    def test_inconsistent_responses_stop_and_preserve_both_observations(self):
        with patch.object(study.cartesia, 'detect', side_effect=[(200, {'detected': True}), (200, {'detected': False})]):
            with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
                study.run(self.plan_path, self.freeze_path, self.directory, 'baseline')
        report = study.load(self.directory/'baseline.json')
        self.assertFalse(report['complete'])
        self.assertEqual([c['detected'] for c in report['checks']], [True, False])
        with patch.object(study.cartesia, 'detect') as detector:
            with self.assertRaisesRegex(ValueError, 'existing phase receipt'):
                study.run(self.plan_path, self.freeze_path, self.directory, 'baseline')
            detector.assert_not_called()

    def test_changed_generation_receipt_is_rejected_before_api(self):
        study.run(self.plan_path, self.freeze_path, self.directory, 'materialize')
        path = self.directory/'generation.json'
        path.write_text(path.read_text()+'\n')
        with patch.object(study.cartesia, 'detect') as detector:
            with self.assertRaisesRegex(ValueError, 'receipt changed'):
                study.run(self.plan_path, self.freeze_path, self.directory, 'baseline')
            detector.assert_not_called()


if __name__ == '__main__':
    unittest.main()
