"""Regression cases for redundant final chunks and standalone fade damage."""
import unittest

import numpy as np

from reconstruction_inference import chunk_starts, render_chunks


class ChunkBoundaryTests(unittest.TestCase):
    def test_recording_covered_by_one_chunk_is_not_split_or_faded(self):
        for length in (1, 50, 90, 91, 95, 99, 100):
            audio = np.linspace(.1, .9, length)
            calls = []

            def render(segment):
                calls.append(len(segment))
                return segment.copy()

            def merge(*args, **kwargs):
                self.fail('A standalone recording must not receive overlap fades')

            actual = render_chunks(audio, 100, 10, render, merge)
            np.testing.assert_array_equal(audio, actual)
            self.assertEqual(calls, [length])

    def test_no_tiny_extra_chunk_after_the_end_is_already_covered(self):
        self.assertEqual(list(chunk_starts(185, 100, 10)), [0, 90])
        self.assertEqual(list(chunk_starts(271, 100, 10)), [0, 90, 180])
        self.assertEqual(list(chunk_starts(191, 100, 10)), [0, 90, 180])

    def test_scheduling_covers_every_sample_with_only_the_last_chunk_partial(self):
        for length in range(1, 1200):
            starts = list(chunk_starts(length, 100, 10))
            covered = np.zeros(length, dtype=bool)
            for index, start in enumerate(starts):
                end = min(length, start+100)
                covered[start:end] = True
                if end == length:
                    self.assertEqual(index, len(starts)-1)
            self.assertTrue(covered.all())

    def test_multi_chunk_merge_gets_native_positions_and_exact_length(self):
        audio = np.arange(185)

        def merge(chunks, chunk_length, hop_length, length):
            self.assertEqual([len(chunk) for chunk in chunks], [100, 95])
            self.assertEqual((chunk_length, hop_length, length), (100, 90, 185))
            np.testing.assert_array_equal(chunks[1], audio[90:])
            return audio.copy()

        result = render_chunks(audio, 100, 10, lambda part: part.copy(), merge)
        np.testing.assert_array_equal(result, audio)

    def test_invalid_schedules_are_rejected(self):
        for args in ((0, 100, 10), (10, 0, 0), (10, 100, -1), (10, 100, 100)):
            with self.assertRaises(ValueError):
                list(chunk_starts(*args))


if __name__ == '__main__':
    unittest.main()
