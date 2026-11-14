import unittest

from experiments.codec_evaluation_text import evaluation_units


class CodecEvaluationTextTests(unittest.TestCase):
    def test_chinese_uses_one_script_before_character_comparison(self):
        calls = []

        def converter(text):
            calls.append(text)
            return text.translate(str.maketrans({'覽': '览', '開': '开'}))

        self.assertEqual(evaluation_units('展覽 開放', 'zh', str.lower, converter), list('展览开放'))
        self.assertEqual(calls, ['展覽 開放'])

    def test_other_languages_keep_their_existing_units_and_unicode_normalization(self):
        def unexpected(text):
            self.fail('Chinese conversion applied to another language')

        self.assertEqual(evaluation_units('展覽 開放', 'ja', str.lower, unexpected), list('展覽開放'))
        self.assertEqual(evaluation_units('ＨＥＬＬＯ World', 'en', str.lower, unexpected), ['hello', 'world'])
        self.assertEqual(evaluation_units('두 단어', 'ko', str.lower, unexpected), ['두', '단어'])


if __name__ == '__main__':
    unittest.main()
