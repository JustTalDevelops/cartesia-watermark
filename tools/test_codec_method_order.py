import unittest

from codec_profiles import method_order


class ProspectiveMethodOrder(unittest.TestCase):
    def test_known_language_profiles_use_fixed_method_orders(self):
        self.assertEqual(method_order('ta', 26.48), ('codec-band-prior', 'codec-band-inverse6-narrow'))
        self.assertEqual(method_order('ko', 29.92)[0], 'codec-band-inverse6')
        self.assertEqual(method_order('ru', 32.88)[0], 'codec-band-prior-repair')
        self.assertEqual(method_order('fr', 30)[0], 'codec-repair')
        self.assertEqual(method_order('ar', 31.68), ('codec-native', 'codec-band-inverse6-consensus'))

    def test_duration_changes_only_the_declared_long_english_profile(self):
        self.assertEqual(method_order('en', 60), ('codec-native', 'codec-band'))
        self.assertEqual(method_order('en', 60.001)[0], 'codec-cycle')
        self.assertEqual(method_order('ta', 101), method_order('ta', 26.48))

    def test_unknown_language_invalid_duration_and_nonfinite_values_reject(self):
        for language, duration in [('xx', 30), ('en', 4), ('en', 301), ('en', float('nan'))]:
            with self.subTest(language=language, duration=duration), self.assertRaises(ValueError):
                method_order(language, duration)


if __name__ == '__main__':
    unittest.main()
