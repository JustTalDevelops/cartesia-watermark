import unittest

from experiments.korean_spacing_diagnostic import spacing_normalized_texts
from speech_repair import disagreement


class KoreanSpacingDiagnosticTests(unittest.TestCase):
    def test_spacing_only_changes_are_separated_from_lexical_errors(self):
        before, after = spacing_normalized_texts('준비해 주세요', '준비해주세요')
        self.assertEqual(before, after)
        self.assertEqual(disagreement(before.split(), after.split()), 0)

    def test_insertions_and_substitutions_cannot_disappear(self):
        for hypothesis in ('준비해 주세오', '준비해 절대 주세요', '아무 말'):
            before, after = spacing_normalized_texts('준비해 주세요', hypothesis)
            self.assertEqual(''.join(after.split()), ''.join(hypothesis.split()))
            self.assertGreater(disagreement(before.split(), after.split()), 0)

    def test_deletion_is_still_counted(self):
        before, after = spacing_normalized_texts('one two three', 'one three')
        self.assertEqual(disagreement(before.split(), after.split()), 1/3)

    def test_existing_inserted_word_boundaries_and_empty_inputs_are_retained(self):
        before, after = spacing_normalized_texts('', 'new words')
        self.assertEqual((before, after), ('', 'new words'))
        self.assertEqual(spacing_normalized_texts('', ''), ('', ''))


if __name__ == '__main__':
    unittest.main()
