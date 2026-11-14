import unittest

from experiments.codec_consensus_guard import supported_substitutions


class ConsensusDiagnosticTests(unittest.TestCase):
    def test_isolated_substitution_needs_support_in_both_anchor_transcripts(self):
        before = 'please bring the blue folder to my desk today'.split()
        after = 'please bring the green folder to my desk today'.split()
        self.assertEqual([v['before_index'] for v in supported_substitutions(before, after, after, after)], [3])
        self.assertEqual(supported_substitutions(before, after, before, after), [])

    def test_inserted_and_deleted_words_are_never_exempted(self):
        before = 'please bring the folder to my desk today'.split()
        after = 'please bring the green folder to my desk today'.split()
        self.assertEqual(supported_substitutions(before, after, after, after), [])
        self.assertEqual(supported_substitutions(after, before, before, before), [])

    def test_repeated_target_phrase_is_ambiguous(self):
        before = 'please bring the blue folder to my desk today'.split()
        after = 'please bring the green folder to my desk today'.split()
        self.assertEqual(supported_substitutions(before, after, after+after, after+after), [])

    def test_a_second_nearby_change_invalidates_the_context(self):
        before = 'please bring the blue folder to my desk today'.split()
        after = 'please bring the green folder at my desk today'.split()
        self.assertEqual(supported_substitutions(before, after, after, after), [])

    def test_short_or_empty_sequences_have_no_supported_substitution(self):
        self.assertEqual(supported_substitutions(['blue'], ['green'], ['green'], ['green']), [])
        self.assertEqual(supported_substitutions([], [], [], []), [])

    def test_both_old_and_new_phrases_in_anchor_are_ambiguous(self):
        before = 'please bring the blue folder to my desk today'.split()
        after = 'please bring the green folder to my desk today'.split()
        self.assertEqual(supported_substitutions(before, after, before+after, before+after), [])


if __name__ == '__main__':
    unittest.main()
