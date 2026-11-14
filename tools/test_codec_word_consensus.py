import unittest

from experiments.codec_word_consensus import comparison_reference


class WordConsensusTests(unittest.TestCase):
    def fixture(self):
        before = [f'w{i}' for i in range(71)]
        after = before.copy()
        for index in (14, 35, 55):
            after[index] = f'new{index}'
        anchor = before.copy()
        anchor[14] = after[14]
        return before, after, anchor

    def test_only_jointly_supported_isolated_correction_is_applied(self):
        before, after, anchor = self.fixture()
        result, proof = comparison_reference(before, after, anchor, anchor)
        self.assertEqual(result[14], after[14])
        self.assertEqual(result[35], before[35])
        self.assertEqual(result[55], before[55])
        self.assertEqual(len(proof), 1)
        self.assertEqual(before[14], 'w14')

    def test_one_changed_anchor_cancels_support(self):
        before, after, anchor = self.fixture()
        changed = anchor.copy()
        changed[14] = 'different'
        result, proof = comparison_reference(before, after, anchor, changed)
        self.assertEqual(result, before)
        self.assertFalse(proof)

    def test_more_than_two_corrections_are_not_partially_cherry_picked(self):
        before, after, _ = self.fixture()
        result, proof = comparison_reference(before, after, after, after)
        self.assertEqual(result, before)
        self.assertFalse(proof)

    def test_correction_fraction_is_bounded_on_short_sources(self):
        before = [f'w{i}' for i in range(20)]
        after = before.copy()
        after[7] = 'new'
        result, proof = comparison_reference(before, after, after, after)
        self.assertEqual(result, before)
        self.assertFalse(proof)

    def test_insertions_and_deletions_are_never_corrections(self):
        before, _, _ = self.fixture()
        for after in (before[:14]+['new']+before[14:], before[:14]+before[15:]):
            with self.subTest(length=len(after)):
                result, proof = comparison_reference(before, after, after, after)
                self.assertEqual(result, before)
                self.assertFalse(proof)


if __name__ == '__main__':
    unittest.main()
