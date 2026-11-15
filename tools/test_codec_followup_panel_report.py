"""Exercise qualification and unrendered rows with mocked HTTP and no models."""
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from experiments import check_codec_frames as common
from experiments import codec_followup_panel_report as reporting
from experiments.codec_profile_cache import store_recognition
from codec_content import assess


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')
    return path


class FollowupQualityTests(unittest.TestCase):
    @contextmanager
    def fixture(self, detected):
        with TemporaryDirectory() as folder:
            root = Path(folder); F = root/'codec_profiles_fresh'; panel = root/'xcodec2_development'
            panel.mkdir(); F.mkdir()
            source, output = root/'source.flac', panel/'candidate.flac'
            source.write_bytes(b'source fixture'); output.write_bytes(b'candidate fixture')
            source_sha, output_sha = common.digest(source), common.digest(output)
            recognition = {'language': 'en', 'primary': 'turbo', 'secondary': 'sensevoice',
                           'turbo': ['one', 'two'], 'sensevoice': ['one', 'two']}
            store_recognition(F/'recognition/originals', source_sha, 'identity', recognition)
            store_recognition(panel/'recognition/processed', output_sha, 'identity', recognition)
            clip = {'source_sha256': source_sha, 'output_sha256': output_sha,
                    'original_dnsmos': {'overall': 4.}, 'speaker_minimum': .98,
                    'dnsmos_delta': {'overall': -.01}}
            receipt = write(panel/'runtime_voice/case0'/(output_sha+'.json'),
                            {'complete': True, 'metric': {'fixture': True}, 'clips': [clip]})
            quality = {'receipt_sha256': common.digest(receipt), 'speaker_minimum': .98,
                       'dnsmos_overall_delta': -.01}
            grid = {'samples': 80000, 'sample_rate': 16000, 'channels': 1}
            old = {'id': 'case0', 'language': 'en', 'source_sha256': source_sha, 'source_format': grid,
                   'original_guard_calibration': {'original_reference_error': {'turbo': 0., 'sensevoice': 0.},
                                                  'language_matches_request': True}}
            authored = write(F/'evaluation.plan.json', {'items': [{'id': 'case0', 'text': 'one two', 'language': 'en'}]})
            study = write(F/'curated_complete_report.json', {'cases': [old], 'endpoint_receipts': {
                'baseline': {'plan_sha256': common.digest(authored)}}})
            write(F/'original_evaluation.json', {'checkpoints': {'small': {'fixture': True}},
                'clips': [{'id': 'case0', 'metric': 'small', 'original': {'reference_error': 0.}}]})
            write(F/'original_voice.json', {'metric': {'fixture': True}, 'clips': [
                {'id': 'case0', 'original_dnsmos': {'overall': 4.}}]})
            generation = write(panel/'generation.json', {'recognition_configuration_sha256': 'identity',
                'cases': [{'id': 'case0', 'speech_assessment': assess(recognition, recognition),
                           'runtime_quality_guard': quality}]})
            write(root/'ticodec_cached_repair_assessment/assessment.json', {'cases': [{'id': 'case0', 'proposals': [
                {'donor': 'native-zero', 'speech_assessment': {'content_guard_passed': False}},
                {'donor': 'continuous-local'}]}]})
            write(root/'ticodec_development/generation.json', {'codec': {'fixture': True}})
            rows = [{'id': 'case0', 'method': 'xcodec2-800bps', 'sha256': output_sha, 'file': str(output),
                     'processing_status': 'processing_eligible', 'endpoint_phase_status': 'pending', 'detector_eligible': True},
                    {'id': 'case0', 'method': 'ticodec-repair-native-zero', 'sha256': 'rejected',
                     'processing_status': 'content_rejected', 'endpoint_phase_status': 'processing_rejected', 'detector_eligible': False},
                    {'id': 'case0', 'method': 'ticodec-repair-continuous-local',
                     'processing_status': 'no_compatible_mms_alignment', 'endpoint_phase_status': 'not_rendered', 'detector_eligible': False}]
            inventory = {'generation_file': str(generation), 'generation_sha256': common.digest(generation),
                'completed_study_file': str(study), 'completed_study_sha256': common.digest(study),
                'code_files_sha256': {}, 'assessment_files_sha256': {},
                'sources': [{'id': 'case0', 'file': str(source), 'sha256': source_sha}], 'treatments': rows,
                'historical_exact_file_observations': {}, 'maximum_http_attempts': 6,
                'planned_sources': 1, 'eligible_treatments': 1, 'repair_assessment_sha256': 'fixture'}
            declaration = write(root/'declaration.json', {**inventory, 'declared_at': common.now()})
            detector = root/'detector.json'; replies = deque([True, True, detected, detected, True, True])
            post = Mock(side_effect=lambda *a, **k: Mock(status_code=200, json=lambda: {'detected': replies.popleft()}))
            with patch.object(reporting, 'data', return_value=inventory), \
                    patch.object(common.cartesia, 'ROOT', root), \
                    patch.object(common.cartesia.requests, 'post', post), \
                    patch.object(common.cartesia, '_pace_detector', return_value=None):
                common.run(declaration, detector, builder=reporting.data)
                measured = common.json.loads(detector.read_text())
                start = datetime.fromisoformat(common.json.loads(declaration.read_text())['declared_at'])
                for index, attempt in enumerate(measured['http_attempts']):
                    attempt['started_at'] = (start+timedelta(seconds=1+80*index)).isoformat()
                write(detector, measured)
                plan = reporting.evaluation_plan(declaration, detector)
                self.assertEqual(len(plan['items']), 1)
                plan_path = write(root/'quality/evaluation.plan.json', plan)
                item = plan['items'][0]
                evaluation = write(root/'quality/evaluation.json', {'complete': True, 'originals_only': False,
                    'candidate_outputs_read': True, 'plan_sha256': common.digest(plan_path),
                    'checkpoints': {'small': {'fixture': True}}, 'clips': [{'id': item['id'], 'metric': 'small',
                    'status': 'scored', 'source_sha256': source_sha, 'output_sha256': output_sha,
                    'original': {'reference_error': 0.}, 'processed': {'reference_error': 0.}, 'reference_error_delta': 0.}]})
                with patch.object(reporting, 'xcodec_export', return_value={'model': {'fixture': True},
                        'cases': [{'id': 'case0', 'channel_parity_controls': []}]}):
                    yield declaration, detector, evaluation, root
                self.assertEqual(post.call_count, 6)

    def test_positive_and_negative_outputs_both_receive_quality_scoring(self):
        for detected in (False, True):
            with self.subTest(detected=detected), self.fixture(detected) as (declaration, detector, evaluation, root):
                result = reporting.export(declaration, detector, evaluation)
                self.assertEqual(result['qualified_spent_sources'], int(not detected))
                self.assertEqual(len(result['treatments']), 3)
                self.assertIsNone(result['treatments'][2]['sha256'])
                self.assertFalse(result['treatments'][2]['qualified_spent_transition'])
                self.assertTrue(result['treatments'][0]['quality_qualified'])
                self.assertNotIn(str(root), json.dumps(result))

    def test_negative_reply_with_missing_quality_evidence_cannot_qualify(self):
        with self.fixture(False) as (declaration, detector, evaluation, _):
            value = json.loads(evaluation.read_text()); value['clips'] = []
            write(evaluation, value)
            with self.assertRaisesRegex(ValueError, 'every declared reference measurement'):
                reporting.export(declaration, detector, evaluation)


if __name__ == '__main__': unittest.main()
