from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.codec_profile_evaluation import evaluation_metrics, limits, reference_score


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


if __name__ == '__main__':
    unittest.main()
