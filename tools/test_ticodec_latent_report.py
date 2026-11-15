"""End-to-end receipt/qualification fixtures; no remote service or neural models."""
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from experiments import check_codec_frames as common
from experiments import check_ticodec_latents as endpoint
from experiments import codec_profile_cache as cache
from experiments import ticodec_latent_report as reporting
from tools import test_ticodec_latent_endpoint as endpoint_tests


class LatentQualityReportTests(unittest.TestCase):
    def fixture(self, folder):
        root, generation_helper = endpoint_tests.LatentEndpointTests().fixture(folder)
        F = root/'codec_profiles_fresh'
        authored = F/'evaluation.plan.json'
        authored.write_text(json.dumps({'items': [{'id': 'case0', 'language': 'en', 'text': 'one two'}]}))
        study_path = F/'curated_complete_report.json'
        study = endpoint.load(study_path)
        study['endpoint_receipts'] = {'baseline': {'plan_sha256': endpoint.digest(authored)}}
        study['cases'][0]['original_guard_calibration'].update(
            original_reference_error={'turbo': 0.0, 'sensevoice': 0.0}, language_matches_request=True)
        study_path.write_text(json.dumps(study))
        manifest = root/'ticodec_latent_development/generation.json'
        generated = endpoint.load(manifest)
        generated['completed_study_sha256'] = endpoint.digest(study_path)
        manifest.write_text(json.dumps(generated))
        original = {'reference_error': 0.0}
        (F/'original_evaluation.json').write_text(json.dumps({'checkpoints': {'small': {'fixture': True}},
            'clips': [{'id': 'case0', 'metric': 'small', 'original': original}]}))
        return root, generation_helper

    @contextmanager
    def evidence(self, helper):
        pilot = {'qualified_spent_transition': True, 'candidate_sha256': 'different-pilot',
                 'case_id': 'different-case', 'source_sha256': 'different-source'}
        def read(folder, checksum, identity):
            return endpoint.load(folder/(checksum+'.json'))
        with helper.evidence(), patch.object(common, 'previous_observations', return_value={}), \
                patch.object(endpoint, 'pilot_export', return_value=pilot), \
                patch.object(cache, 'read_recognition', side_effect=read):
            yield

    def measure(self, root, detected):
        declaration, detector = root/'declaration.json', root/'detector.json'
        endpoint.declare(root/'ticodec_latent_development/generation.json',
                         root/'codec_profiles_fresh/curated_complete_report.json', declaration)
        replies = deque([True, True, detected, detected, True, True])
        post = Mock(side_effect=lambda *a, **k: Mock(status_code=200, json=lambda: {'detected': replies.popleft()}))
        with patch.object(common.cartesia.requests, 'post', post), patch.object(common.cartesia, 'ROOT', root), \
                patch.object(common.cartesia, '_pace_detector', return_value=None):
            common.run(declaration, detector, builder=endpoint.data)
        measured = endpoint.load(detector)
        start = datetime.fromisoformat(endpoint.load(declaration)['declared_at'])
        for index, attempt in enumerate(measured['http_attempts']):
            attempt['started_at'] = (start+timedelta(seconds=1+80*index)).isoformat()
        detector.write_text(json.dumps(measured))
        plan = reporting.evaluation_plan(declaration, detector)
        self.assertEqual(len(plan['items']), 1)
        self.assertEqual(post.call_count, 6)
        quality = root/'quality'; quality.mkdir()
        plan_path = quality/'evaluation.plan.json'; plan_path.write_text(json.dumps(plan))
        item = plan['items'][0]
        evaluated = {'complete': True, 'originals_only': False, 'candidate_outputs_read': True,
            'plan_sha256': endpoint.digest(plan_path), 'checkpoints': {'small': {'fixture': True}},
            'clips': [{'id': item['id'], 'metric': 'small', 'status': 'scored',
                       'source_sha256': item['sha256'], 'output_sha256': endpoint.digest(Path(item['output'])),
                       'original': {'reference_error': 0.0}, 'processed': {'reference_error': 0.0},
                       'reference_error_delta': 0.0}]}
        evaluation = quality/'evaluation.json'; evaluation.write_text(json.dumps(evaluated))
        return declaration, detector, evaluation

    def test_negative_and_positive_files_both_get_quality_evaluation_and_all_rejections_remain(self):
        for detected in (False, True):
            with self.subTest(detected=detected), TemporaryDirectory() as folder:
                root, helper = self.fixture(folder)
                with self.evidence(helper):
                    declaration, detector, evaluation = self.measure(root, detected)
                    result = reporting.export(declaration, detector, evaluation)
                self.assertEqual(result['qualified_spent_sources'], int(not detected))
                self.assertEqual(len(result['treatments']), 3)
                self.assertEqual(len(result['exact_quantized_controls']), 1)
                self.assertTrue(result['treatments'][0]['quality_qualified'])
                self.assertEqual(result['fresh_efficacy_sources'], 0)
                self.assertFalse(result['completed_19_of_44_result_changed'])
                self.assertNotIn(str(root), json.dumps(result))

    def test_missing_reference_measurement_cannot_qualify_a_negative_file(self):
        with TemporaryDirectory() as folder:
            root, helper = self.fixture(folder)
            with self.evidence(helper):
                declaration, detector, evaluation = self.measure(root, False)
                value = endpoint.load(evaluation); value['clips'] = []
                evaluation.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'every declared reference measurement'):
                    reporting.export(declaration, detector, evaluation)


if __name__ == '__main__': unittest.main()
