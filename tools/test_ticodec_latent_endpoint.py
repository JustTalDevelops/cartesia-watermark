"""Validate phase dependencies and fixed latent inventories without API calls."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import check_ticodec_latents as endpoint
from experiments import ticodec_latent_generation_report as reporting
from tools import test_ticodec_latent_reporting as generation_tests


class LatentEndpointTests(unittest.TestCase):
    def fixture(self, folder):
        helper = generation_tests.LatentReportingTests()
        root, _, _ = helper.fixture(folder)
        for name in ('codec_frame_development', 'ticodec_joint'):
            panel = root/name
            panel.mkdir(exist_ok=True)
            source = {'id': 'old-source', 'sha256': 'old-source-hash', 'file': '/offline/source.flac'}
            candidate = {'id': 'old-source', 'sha256': 'old-candidate-hash', 'file': '/offline/candidate.flac',
                         'detector_eligible': True, 'method': 'old-method'}
            declared = {'maximum_http_attempts': 6, 'sources': [source], 'treatments': [candidate],
                        'declared_at': '2026-01-01T00:00:00+00:00'}
            declaration = panel/'endpoint.declaration.json'
            declaration.write_text(json.dumps(declared))
            checks, attempts = [], []
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            for role, row in [('opening_positive_control', source), ('candidate', candidate),
                              ('closing_positive_control', source)]:
                for ordinal in (1, 2):
                    item = {k: row[k] for k in ('id', 'sha256', 'file')}
                    if 'method' in row: item['method'] = row['method']
                    item.update(role=role, ordinal=ordinal, http_status=200)
                    attempts.append(dict(item, started_at=(start+timedelta(seconds=1+80*len(checks))).isoformat()))
                    checks.append(dict(item, detected=role != 'candidate'))
            (panel/'detector.json').write_text(json.dumps({'complete': True, 'checks': checks,
                'http_attempts': attempts, 'declaration_sha256': endpoint.digest(declaration)}))
        pilot = root/'ticodec_pilot'
        (pilot/'quality').mkdir(parents=True)
        for path in (root/'ticodec_pilot.declaration.json', pilot/'quality/evaluation.json',
                     pilot/'quality/evaluation.plan.json'):
            path.write_text('{}')
        (pilot/'result.json').write_text('{"checks":[]}')
        return root, helper

    def test_all_latent_rows_remain_and_only_processing_eligible_files_are_queried(self):
        with TemporaryDirectory() as folder:
            root, helper = self.fixture(folder)
            pilot = {'qualified_spent_transition': True, 'candidate_sha256': 'different-pilot',
                     'case_id': 'prior-pilot', 'source_sha256': 'prior-source'}
            with helper.evidence(), patch.object(endpoint.endpoint, 'previous_observations', return_value={}), \
                    patch.object(endpoint, 'pilot_export', return_value=pilot):
                result = endpoint.data(root/'ticodec_latent_development/generation.json',
                                       root/'codec_profiles_fresh/curated_complete_report.json')
            self.assertEqual(len(result['treatments']), 3)
            self.assertEqual(result['eligible_treatments'], 1)
            self.assertEqual(result['maximum_http_attempts'], 6)
            self.assertEqual(result['historical_exact_file_observations'],
                             {'old-source-hash': True, 'old-candidate-hash': False})

    def test_incomplete_earlier_phase_stops_before_inventory_or_transport(self):
        with TemporaryDirectory() as folder:
            root, _ = self.fixture(folder)
            path = root/'ticodec_joint/detector.json'
            value = endpoint.load(path); value['complete'] = False
            path.write_text(json.dumps(value))
            with patch.object(endpoint.endpoint, 'previous_observations', return_value={}), \
                    patch.object(endpoint, 'validated_inventory') as inventory, \
                    patch.object(endpoint.endpoint.cartesia.requests, 'post') as post:
                with self.assertRaisesRegex(ValueError, 'both preceding endpoint'):
                    endpoint.data(root/'ticodec_latent_development/generation.json',
                                  root/'codec_profiles_fresh/curated_complete_report.json')
            inventory.assert_not_called(); post.assert_not_called()

    def test_conflicting_historical_reply_stops_a_new_phase(self):
        with TemporaryDirectory() as folder:
            root, _ = self.fixture(folder)
            path = root/'ticodec_joint/detector.json'
            value = endpoint.load(path); value['checks'][3]['detected'] = True
            path.write_text(json.dumps(value))
            with patch.object(endpoint.endpoint, 'previous_observations', return_value={}):
                with self.assertRaisesRegex(ValueError, 'replies conflict'):
                    endpoint.previous_receipts(root, {})

    def test_prior_qualified_exact_file_reuses_its_original_pilot_receipts(self):
        with TemporaryDirectory() as folder:
            root, helper = self.fixture(folder)
            with helper.evidence(): public, _ = reporting.validated_inventory(root)
            case = public['cases'][0]
            pilot = {'qualified_spent_transition': True, 'candidate_sha256': case['treatments'][0]['output_sha256'],
                     'case_id': case['id'], 'source_sha256': case['source_sha256'], 'endpoint_sha256': 'prior-receipts'}
            with helper.evidence(), patch.object(endpoint.endpoint, 'previous_observations', return_value={}), \
                    patch.object(endpoint, 'pilot_export', return_value=pilot):
                result = endpoint.data(root/'ticodec_latent_development/generation.json',
                                       root/'codec_profiles_fresh/curated_complete_report.json')
            self.assertEqual(result['eligible_treatments'], 0)
            self.assertEqual(result['maximum_http_attempts'], 4)
            self.assertEqual(result['treatments'][0]['endpoint_phase_status'], 'qualified_pilot_reused')


if __name__ == '__main__': unittest.main()
