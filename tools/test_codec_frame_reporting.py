"""Offline regression checks for the all-eligible frame evaluation inventory."""
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import test_codec_frame_endpoint as endpoint_tests
from experiments import codec_frame_report as reporting
from experiments import check_codec_frames as checker


class FrameReportingTests(unittest.TestCase):
    def fixture(self, folder):
        helper = endpoint_tests.FrameEndpointTests()
        value, declaration, output = helper.fixture(folder)
        root = Path(folder)
        authored = root/'evaluation.plan.json'
        authored.write_text(json.dumps({'items':[{'id':'case0','source':value['sources'][0]['file'],
            'sha256':value['sources'][0]['sha256'], 'language':'en', 'text':'Offline test reference.'}]}))
        study = Path(value['completed_study_file'])
        study.write_text(json.dumps({'endpoint_receipts':{'baseline':{'plan_sha256':checker.digest(authored)}}}))
        value['completed_study_sha256'] = checker.digest(study)
        declaration.write_text(json.dumps(dict(value,declared_at=checker.now())))
        helper.execute(value,declaration,output,[True]*6)
        report = json.loads(output.read_text())
        start = datetime.fromisoformat(json.loads(declaration.read_text())['declared_at'])
        for index,row in enumerate(report['http_attempts']):
            row['started_at'] = (start+timedelta(seconds=1+80*index)).isoformat()
        output.write_text(json.dumps(report))
        return value,declaration,output

    def test_all_eligible_treatments_are_scored_even_when_positive_or_duplicate(self):
        with TemporaryDirectory() as folder:
            value,declaration,output = self.fixture(folder)
            with patch.object(reporting,'declaration_data',return_value=value):
                plan = reporting.evaluation_plan(declaration,output)
            self.assertEqual([r['id'] for r in plan['items']],['case0__offset80','case0__offset160'])
            self.assertEqual(len({r['output'] for r in plan['items']}),1)
            self.assertTrue(plan['all_eligible_including_detected'])

    def test_incomplete_endpoint_blocks_reference_evaluation(self):
        with TemporaryDirectory() as folder:
            value,declaration,output = self.fixture(folder)
            result = json.loads(output.read_text());result['complete']=False
            output.write_text(json.dumps(result))
            with patch.object(reporting,'declaration_data',return_value=value):
                with self.assertRaisesRegex(ValueError,'Complete the declared detector'):
                    reporting.evaluation_plan(declaration,output)

    def test_missing_repeat_or_changed_pacing_cannot_be_curated(self):
        with TemporaryDirectory() as folder:
            value,declaration,output = self.fixture(folder)
            original = json.loads(output.read_text())
            result = json.loads(output.read_text());result['checks'].pop()
            output.write_text(json.dumps(result))
            with patch.object(reporting,'declaration_data',return_value=value):
                with self.assertRaisesRegex(ValueError,'actual request'):
                    reporting.validated_detector(declaration,output)
                result=original
                result['http_attempts'][1]['started_at']=result['http_attempts'][0]['started_at']
                output.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'pacing'):
                    reporting.validated_detector(declaration,output)

    def test_incomplete_generation_cannot_declare_or_send_any_request(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);generation=root/'generation.json';study=root/'study.json'
            generation.write_text('{"complete":false}');study.write_text('{}')
            with patch.object(checker.cartesia.requests,'post') as post:
                with self.assertRaisesRegex(ValueError,'completed generation'):
                    checker.declare(generation,study,root/'declaration.json')
            post.assert_not_called()
            self.assertFalse((root/'declaration.json').exists())


if __name__=='__main__':
    unittest.main()
