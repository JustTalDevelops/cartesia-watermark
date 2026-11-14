import unittest

from codec_content import assess, secondary_for
from codec_text import paired_error, text_units


class CodecProfileTests(unittest.TestCase):
    def test_hindi_vowel_signs_and_virama_remain_in_words(self):
        self.assertEqual(text_units('किताब, कृपया।', str.lower, 'hi'), list('किताबकृपया'))
        self.assertNotEqual(text_units('दिन', str.lower, 'hi'), text_units('दीन', str.lower, 'hi'))

    def test_tamil_vowel_signs_remain_in_words(self):
        self.assertEqual(text_units('தமிழ் மொழி!', str.lower, 'ta'), list('தமிழ்மொழி'))
        self.assertNotEqual(text_units('கல்', str.lower, 'ta'), text_units('கால்', str.lower, 'ta'))

    def test_korean_spacing_does_not_create_a_lexical_disagreement(self):
        before = {'language': 'ko', 'secondary': 'sensevoice',
                  'turbo': ['준비해', '주세요'], 'sensevoice': ['준비해', '주세요']}
        after = {'language': 'ko', 'secondary': 'sensevoice',
                 'turbo': ['준비해주세요'], 'sensevoice': ['준비해주세요']}
        result = assess(before, after)
        self.assertTrue(result['content_guard_passed'])
        self.assertEqual(result['paired_disagreement'], {'turbo': 0., 'sensevoice': 0.})
        self.assertNotIn('mms', result['paired_disagreement'])

    def test_indic_character_limit_is_one_percent_and_reports_actual_primary(self):
        before = {'language': 'hi', 'primary': 'large', 'secondary': 'mms', 'large': ['क']*100, 'mms': ['क']*100}
        changed = ['क']*100
        changed[50] = 'ख'
        after = dict(before, large=changed)
        report = assess(before, after)
        self.assertTrue(report['content_guard_passed'])
        self.assertEqual(report['maximum_paired_disagreement'], .01)
        self.assertNotIn('turbo', report['paired_disagreement'])
        changed[51] = 'ख'
        self.assertFalse(assess(before, after)['content_guard_passed'])

    def test_arabic_profile_uses_the_pinned_fastconformer_export(self):
        self.assertEqual(secondary_for('ar'), 'fastconformer')
        before = {'language': 'ar', 'secondary': 'fastconformer', 'turbo': ['عمل'], 'fastconformer': ['عمل']}
        self.assertTrue(assess(before, before)['content_guard_passed'])

    def test_primary_checkpoint_cannot_change_during_processing(self):
        before = {'language': 'hi', 'primary': 'large', 'secondary': 'mms', 'large': ['क'], 'mms': ['क']}
        self.assertFalse(assess(before, dict(before, primary='turbo', turbo=['क']))['content_guard_passed'])

    def test_actual_inserted_words_still_reject(self):
        self.assertGreater(paired_error(['준비해', '주세요'], ['준비해', '절대', '주세요'], 'ko'), .03)

    def test_secondary_model_cannot_change_between_original_and_output(self):
        before = {'language': 'en', 'secondary': 'sensevoice', 'turbo': ['one'], 'sensevoice': ['one']}
        after = {'language': 'en', 'secondary': 'mms', 'turbo': ['one'], 'mms': ['one'], 'sensevoice': ['one']}
        self.assertFalse(assess(before, after)['content_guard_passed'])

    def test_chinese_conversion_does_not_apply_to_japanese(self):
        convert = lambda text: text.replace('設計', '设计')
        self.assertEqual(text_units('設計', str.lower, 'zh', converter=convert), ['设', '计'])
        self.assertEqual(text_units('設計', str.lower, 'ja', converter=convert), ['設', '計'])

    def test_empty_secondary_speech_is_not_a_passing_comparison(self):
        before = {'language': 'en', 'secondary': 'sensevoice', 'turbo': ['one'], 'sensevoice': ['one']}
        after = dict(before, sensevoice=[])
        self.assertFalse(assess(before, after)['content_guard_passed'])


if __name__ == '__main__':
    unittest.main()
