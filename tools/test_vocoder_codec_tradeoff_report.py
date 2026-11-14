"""A negative detector result cannot dispense with its source controls."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('tradeoff_report', ROOT / 'experiments/vocoder_codec_tradeoff_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class EndpointEvidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.assessment = self.folder / 'assessment.json'
        self.assessment.write_text('{}\n')
        started = datetime(2026, 11, 14, tzinfo=timezone.utc)
        roles = ['opening_original'] * 2 + ['candidate'] * 2 + ['closing_original'] * 2
        self.evidence = {'complete': True, 'source_sha256': 'source', 'candidate_sha256': 'candidate',
                         'assessment_sha256': report.digest(self.assessment), 'candidate_detected': False,
                         'checks': [{'role': role, 'ordinal': i % 2 + 1,
                                     'sha256': 'candidate' if role == 'candidate' else 'source',
                                     'detected': role != 'candidate', 'http_status': 200} for i, role in enumerate(roles)],
                         'http_attempts': [{'role': role, 'ordinal': i % 2 + 1,
                                           'sha256': 'candidate' if role == 'candidate' else 'source',
                                           'http_status': 200,
                                           'started_at': (started + timedelta(seconds=i * 80)).isoformat()} for i, role in enumerate(roles)]}

    def verify(self, evidence):
        (self.folder / 'result.json').write_text(json.dumps(evidence))
        return report.verify_endpoint(self.folder, self.assessment, 'source', 'candidate')

    def test_complete_negative_with_positive_brackets(self):
        self.assertFalse(self.verify(self.evidence)['candidate_detected'])

    def test_missing_or_negative_closing_controls_rejected(self):
        for mutation in ('missing', 'negative'):
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.evidence)
                if mutation == 'missing':
                    changed['checks'].pop()
                else:
                    changed['checks'][-1]['detected'] = False
                with self.assertRaises(ValueError):
                    self.verify(changed)

    def test_replaced_assessment_or_candidate_rejected(self):
        for key in ('assessment_sha256', 'candidate_sha256', 'source_sha256'):
            with self.subTest(key=key):
                changed = deepcopy(self.evidence)
                changed[key] = 'changed'
                with self.assertRaises(ValueError):
                    self.verify(changed)

    def test_inconsistent_candidate_responses_rejected(self):
        changed = deepcopy(self.evidence)
        changed['checks'][3]['detected'] = True
        with self.assertRaises(ValueError):
            self.verify(changed)

    def test_wrong_uploaded_bytes_rejected(self):
        for section in ('checks', 'http_attempts'):
            with self.subTest(section=section):
                changed = deepcopy(self.evidence)
                changed[section][2]['sha256'] = 'other-upload'
                with self.assertRaises(ValueError):
                    self.verify(changed)

    def test_unfinished_or_unpaced_protocol_rejected(self):
        changed = deepcopy(self.evidence)
        changed['complete'] = False
        with self.assertRaises(ValueError):
            self.verify(changed)
        changed = deepcopy(self.evidence)
        changed['http_attempts'][3] = changed['http_attempts'][2]
        with self.assertRaises(ValueError):
            self.verify(changed)


class AdditionalQualityEvidence(unittest.TestCase):
    def test_original_calibration_and_measured_delta_are_both_required(self):
        review = {'original_reference_error': .1, 'original_calibrated': True}
        for delta, passes in ((.02, True), (.04, False)):
            measured = {'processed_reference_error': .1 + delta,
                        'reference_error_delta': delta, 'reference_delta_passed': passes}
            self.assertEqual(report.additional_review_passes(review, measured), passes)
        review = {'original_reference_error': .2, 'original_calibrated': False}
        self.assertFalse(report.additional_review_passes(review, {
            'processed_reference_error': .2, 'reference_error_delta': 0,
            'reference_delta_passed': True}))

    def test_claimed_pass_cannot_override_observed_regression(self):
        with self.assertRaises(ValueError):
            report.additional_review_passes({'original_reference_error': .1, 'original_calibrated': True}, {
                'processed_reference_error': .14, 'reference_error_delta': .04,
                'reference_delta_passed': True})


if __name__ == '__main__':
    unittest.main()
