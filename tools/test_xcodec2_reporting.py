"""Reject incomplete or inconsistent codec evidence before publication."""
from copy import deepcopy
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.xcodec2_generation_report import validate_completion, validate_parity


def fixtures():
    rows = [{'id': str(i), 'status': 'content_rejected', 'source_format': {'channels': 2},
             'channel_parity_controls': [{'channel': c, 'code_indices_exact': True,
                'decoded_waveform_exact': True, 'rng_advance_exact': True,
                'full_seconds': 2., 'prefix_seconds': 1.8} for c in range(2)]}
            for i in range(25)]
    measured = {'complete': True, 'planned_sources': 25, 'planned_treatments': 25,
                'new_tts_requests': 0, 'new_http_requests': 0, 'fresh_efficacy_sources': 0,
                'reference_text_used_for_generation_or_repair': False,
                'independent_quality_evaluated': False, 'detector_checked': False, 'cases': rows}
    declared = {'cases': deepcopy(rows), 'planned_sources': 25, 'planned_treatments': 25}
    study = {'cases': [{'id': str(i), 'marked_provenance': True,
                       'qualified_marked_transition': False} for i in range(25)]}
    return measured, declared, study


class ReportingTests(unittest.TestCase):
    def test_all_rejections_are_retained(self):
        self.assertEqual(len(validate_completion(*fixtures())), 25)

    def test_incomplete_run_cannot_export(self):
        measured, declared, study = fixtures(); measured['complete'] = False
        with self.assertRaises(ValueError): validate_completion(measured, declared, study)

    def test_missing_failure_cannot_export(self):
        measured, declared, study = fixtures(); measured['cases'].pop()
        with self.assertRaises(ValueError): validate_completion(measured, declared, study)

    def test_unfinished_row_cannot_export(self):
        measured, declared, study = fixtures(); measured['cases'][0]['status'] = 'processing'
        with self.assertRaises(ValueError): validate_completion(measured, declared, study)

    def test_duplicate_substitution_cannot_export(self):
        measured, declared, study = fixtures(); measured['cases'][1] = deepcopy(measured['cases'][0])
        with self.assertRaises(ValueError): validate_completion(measured, declared, study)

    def test_every_channel_requires_its_own_control(self):
        measured, _, _ = fixtures(); row = measured['cases'][0]
        row['channel_parity_controls'][1]['channel'] = 0
        with self.assertRaises(ValueError): validate_parity(row)

    def test_waveform_or_codes_or_rng_mismatch_is_rejected(self):
        for key in ('code_indices_exact', 'decoded_waveform_exact', 'rng_advance_exact'):
            with self.subTest(key=key):
                measured, _, _ = fixtures(); row = measured['cases'][0]
                row['channel_parity_controls'][0][key] = False
                with self.assertRaises(ValueError): validate_parity(row)

    def test_numeric_truthiness_does_not_establish_parity(self):
        measured, _, _ = fixtures(); row = measured['cases'][0]
        row['channel_parity_controls'][0]['decoded_waveform_exact'] = 1
        with self.assertRaises(ValueError): validate_parity(row)

    def test_nonfinite_or_invalid_timing_is_rejected(self):
        for value in (float('nan'), float('inf'), 0, -1, True):
            with self.subTest(value=value):
                measured, _, _ = fixtures(); row = measured['cases'][0]
                row['channel_parity_controls'][0]['full_seconds'] = value
                with self.assertRaises(ValueError): validate_parity(row)

    def test_generation_cannot_claim_independent_validation(self):
        for key in ('independent_quality_evaluated', 'detector_checked',
                    'reference_text_used_for_generation_or_repair'):
            with self.subTest(key=key):
                measured, declared, study = fixtures(); measured[key] = True
                with self.assertRaises(ValueError): validate_completion(measured, declared, study)


if __name__ == '__main__': unittest.main()
