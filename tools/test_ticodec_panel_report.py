"""Offline checks for complete joint inventories and unchanged service receipts."""
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import test_ticodec_endpoint as endpoint_tests
from experiments import check_codec_frames as checker
from experiments import ticodec_panel_report as reporting


class JointReportingTests(unittest.TestCase):
    def fixture(self, folder):
        helper = endpoint_tests.TiCodecEndpointTests()
        value, declaration, output = helper.fixture(folder)
        root = Path(folder)
        value['band_code_root'] = str(root)
        prior = value['treatments'][0]
        prior.pop('offset_native_samples')
        prior['method'] = 'full-codec'
        authored = root/'evaluation.plan.json'
        authored.write_text(json.dumps({'items': [{
            'id': 'case0', 'language': 'en', 'text': 'Offline reference.'}]}))
        study = Path(value['completed_study_file'])
        study.write_text(json.dumps({'endpoint_receipts': {
            'baseline': {'plan_sha256': checker.digest(authored)}}}))
        value['completed_study_sha256'] = checker.digest(study)
        declaration.write_text(json.dumps(dict(value, declared_at=checker.now())))
        helper.execute(value, declaration, output, lambda *args: value)
        result = json.loads(output.read_text())
        start = datetime.fromisoformat(json.loads(declaration.read_text())['declared_at'])
        for index, row in enumerate(result['http_attempts']):
            row['started_at'] = (start+timedelta(seconds=1+80*index)).isoformat()
        output.write_text(json.dumps(result))
        return value, declaration, output

    def test_positive_and_duplicate_treatments_are_scored_prior_pilot_is_retained(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            result = json.loads(output.read_text())
            for check in result['checks']:
                if check['role'] == 'candidate': check['detected'] = True
            for observation in result['observations'].values(): observation['detected'] = True
            for row in result['treatments']:
                if row['detector_eligible']: row['detected'] = True
            output.write_text(json.dumps(result))
            with patch.object(reporting, 'data', return_value=value):
                plan = reporting.evaluation_plan(declaration, output)
                _, validated = reporting.validated(declaration, output)
            self.assertEqual([i['id'] for i in plan['items']],
                             ['case0__band-wide', 'case0__band-narrow'])
            self.assertEqual(len({i['output'] for i in plan['items']}), 1)
            self.assertTrue(plan['all_new_eligible_including_detected'])
            self.assertEqual(validated['treatments'][0]['endpoint_phase_status'],
                             'qualified_pilot_reused')

    def test_incomplete_endpoint_cannot_create_reference_plan(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            result = json.loads(output.read_text()); result['complete'] = False
            output.write_text(json.dumps(result))
            with patch.object(reporting, 'data', return_value=value):
                with self.assertRaisesRegex(ValueError, 'entire joint endpoint'):
                    reporting.evaluation_plan(declaration, output)

    def test_missing_repeat_or_changed_pacing_is_rejected(self):
        for mutation, message in [('repeat', 'every actual request'), ('pacing', 'pacing')]:
            with self.subTest(mutation=mutation), TemporaryDirectory() as folder:
                value, declaration, output = self.fixture(folder)
                result = json.loads(output.read_text())
                if mutation == 'repeat': result['checks'].pop()
                else:
                    result['http_attempts'][1]['started_at'] = result['http_attempts'][0]['started_at']
                output.write_text(json.dumps(result))
                with patch.object(reporting, 'data', return_value=value):
                    with self.assertRaisesRegex(ValueError, message):
                        reporting.validated(declaration, output)

    def test_method_identity_and_actual_repeat_replies_must_match(self):
        for mutation, message in [('identity', 'exact-file observations'), ('reply', 'Conflicting')]:
            with self.subTest(mutation=mutation), TemporaryDirectory() as folder:
                value, declaration, output = self.fixture(folder)
                result = json.loads(output.read_text())
                if mutation == 'identity':
                    next(iter(result['observations'].values()))['method'] = 'different-method'
                else: result['checks'][2]['detected'] = True
                output.write_text(json.dumps(result))
                with patch.object(reporting, 'data', return_value=value):
                    with self.assertRaisesRegex(ValueError, message):
                        reporting.validated(declaration, output)

    def test_changed_authored_reference_is_rejected(self):
        with TemporaryDirectory() as folder:
            value, declaration, output = self.fixture(folder)
            (Path(folder)/'evaluation.plan.json').write_text('{"items":[]}')
            with patch.object(reporting, 'data', return_value=value):
                with self.assertRaisesRegex(ValueError, 'Authored references'):
                    reporting.evaluation_plan(declaration, output)


if __name__ == '__main__': unittest.main()
