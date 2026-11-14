import unittest
import numpy as np

from speech_repair import (AlignmentUnavailable, RepairTooLarge, aligned_regions,
                           blend_regions, disagreement, language_from_chunks, pcm_digest, units)


class SpeechRepairTests(unittest.TestCase):
    def test_absent_longform_tags_request_detection_and_missing_tags_do_not_conflict(self):
        codes = {'english': 'en'}
        result = {'chunks': [{'text': 'first', 'language': None}, {'text': 'second'}]}
        self.assertEqual(language_from_chunks(result, codes), (None, True))
        result['chunks'][0]['language'] = 'english'
        self.assertEqual(language_from_chunks(result, codes), ('en', False))

    def test_conflicting_unknown_and_empty_languages_remain_rejected(self):
        codes = {'english': 'en', 'arabic': 'ar'}
        result = {'chunks': [{'text': 'first', 'language': 'english'},
                             {'text': 'second', 'language': 'arabic'}]}
        self.assertEqual(language_from_chunks(result, codes), (None, False))
        self.assertEqual(language_from_chunks({'chunks': [{'text': 'word', 'language': 'unknown'}]}, codes), (None, False))
        self.assertEqual(language_from_chunks({'chunks': [{'text': ' ', 'language': None}]}, codes), (None, False))

    def test_word_and_character_units_and_empty_speech(self):
        self.assertEqual(units('Hello  WORLD', str.lower, 'en'), ['hello', 'world'])
        self.assertEqual(units('你 好', str.lower, 'zh'), ['你', '好'])
        self.assertIsNone(disagreement([], []))
        self.assertIsNone(disagreement(['speech'], []))
        self.assertEqual(disagreement(['one', 'two'], ['one', 'three']), .5)

    def test_alignment_uses_both_timing_intervals(self):
        before = {'text': 'hello world', 'chunks': [{'text': 'hello', 'timestamp': (0, 1)},
                                                   {'text': 'world', 'timestamp': (1, 2)}]}
        after = {'text': 'hello there', 'chunks': [{'text': 'hello', 'timestamp': (0, 1.1)},
                                                  {'text': 'there', 'timestamp': (1.1, 2.1)}]}
        self.assertEqual(aligned_regions(before, after, 3, str.lower, 'en'), [(.92, 2.18)])
        after['chunks'][1]['timestamp'] = (1.1, None)
        with self.assertRaises(AlignmentUnavailable):
            aligned_regions(before, after, 3, str.lower, 'en')

    def test_missing_timestamps_for_unchanged_words_do_not_create_repairs(self):
        result = {'text': 'hello', 'chunks': [{'text': 'hello', 'timestamp': (0, None)}]}
        self.assertEqual(aligned_regions(result, result, 3, str.lower, 'en'), [])
        incomplete = {'text': 'hello world', 'chunks': result['chunks']}
        with self.assertRaises(AlignmentUnavailable):
            aligned_regions(result, incomplete, 3, str.lower, 'en')

    def test_blending_preserves_unselected_samples_and_stereo_balance(self):
        base = np.full((8000, 2), [.1, .05], dtype=np.float32)
        donor = np.full((8000, 2), [.4, .2], dtype=np.float32)
        original = base.copy()
        output, coverage, gain = blend_regions(base, donor, 8000, [(.2, .3)])
        self.assertEqual(gain, 1)
        self.assertLessEqual(coverage, .1)
        np.testing.assert_array_equal(base, original)
        np.testing.assert_array_equal(output[:1600], original[:1600])
        np.testing.assert_array_equal(output[2400:], original[2400:])
        np.testing.assert_array_equal(output[:, 0], output[:, 1]*2)

    def test_large_repair_is_rejected_before_it_can_be_used(self):
        with self.assertRaises(RepairTooLarge):
            blend_regions(np.zeros(8000), np.ones(8000), 8000, [(0, .5)])

    def test_shared_peak_gain_and_pcm_digest_bind_exact_samples(self):
        base = np.full((8000, 2), [1.2, .6], dtype=np.float32)
        output, _, gain = blend_regions(base, base, 8000, [(.2, .3)])
        self.assertLess(gain, 1)
        self.assertLessEqual(np.abs(output).max(), .999001)
        np.testing.assert_array_equal(output[:, 0], output[:, 1]*2)
        self.assertEqual(pcm_digest(output), pcm_digest(output.astype(np.float64)))
        changed = output.copy()
        changed[0, 0] = 0
        self.assertNotEqual(pcm_digest(output), pcm_digest(changed))


if __name__ == '__main__':
    unittest.main()
