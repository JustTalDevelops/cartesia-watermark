from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.codec_profile_evaluation import evaluation_metrics, limits, reference_score, quality_criteria


class ProfileEvaluationPolicy(unittest.TestCase):
    def test_roles_and_limits_are_explicit_before_any_new_source(self):
        self.assertEqual(evaluation_metrics('hi'), ('turbo',))
        self.assertEqual(evaluation_metrics('ar'), ('large', 'mms'))
        self.assertEqual(evaluation_metrics('ta'), ('small',))
        self.assertEqual(limits('ta'), (.01, .10))
        self.assertEqual(limits('en'), (.03, .15))

    def test_tamil_and_hindi_vowel_marks_are_not_erased_by_a_word_normalizer(self):
        for original, changed, language in [('வீடு நூல்', 'விடு நூல்', 'ta'), ('किताब घर', 'कताब घर', 'hi')]:
            same = reference_score(original, original, language, str.lower)
            different = reference_score(original, changed, language, str.lower)
            self.assertEqual(same['reference_error'], 0)
            self.assertGreater(different['reference_error'], 0)
            self.assertGreater(different['raw_word_reference_error_diagnostic_only'], 0)
            self.assertEqual(different['comparison_units'], 'characters')

    def test_english_uses_words_and_empty_recognition_does_not_pass_calibration(self):
        row = reference_score('one two three', 'one wrong three', 'en', str.lower)
        self.assertAlmostEqual(row['reference_error'], 1/3)
        self.assertEqual(row['comparison_units'], 'words')
        empty = reference_score('one two three', '', 'en', str.lower)
        self.assertFalse(empty['calibrated'])

    def tamil_quality(self, **changes):
        values = dict(language='ta', paired={'large': .0072, 'mms': .0023},
                      guard_deltas={'large': .0071, 'mms': .0023}, evaluation_deltas={'small': 0.},
                      original_errors={'large': .024, 'mms': .031, 'small': .077},
                      speaker_minimum=.986, dnsmos_delta=-.074, content_guard_passed=True,
                      language_matches_request=True)
        values.update(changes)
        return quality_criteria(**values)

    def test_independent_speech_failure_is_retained_despite_passing_processing_guards(self):
        self.assertTrue(self.tamil_quality()['quality_qualified'])
        failed = self.tamil_quality(evaluation_deltas={'small': .0215311004784689})
        self.assertFalse(failed['quality_qualified'])
        self.assertFalse(failed['quality_gate_results']['evaluation_small_reference_delta'])

    def test_calibration_language_and_unmeasured_primary_cannot_be_waived(self):
        for changes in ({'original_errors': {'large': .2, 'mms': .2, 'small': .01}},
                        {'language_matches_request': False}, {'paired': {'large': None, 'mms': 0.}}):
            self.assertFalse(self.tamil_quality(**changes)['quality_qualified'])

    def test_arabic_requires_its_extra_independent_checkpoint(self):
        with self.assertRaises(ValueError):
            quality_criteria('ar', {'turbo': 0., 'fastconformer': 0.},
                             {'turbo': 0., 'fastconformer': 0.}, {'large': 0.},
                             {'turbo': .1, 'fastconformer': .1, 'large': .1}, .99, 0., True, True)


if __name__ == '__main__':
    unittest.main()
