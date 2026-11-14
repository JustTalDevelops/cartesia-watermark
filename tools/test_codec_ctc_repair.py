import unittest

import numpy as np

from experiments.codec_ctc_repair import repair_from_native
from speech_repair import AlignmentUnavailable, RepairTooLarge


class CodecCTCRepairTests(unittest.TestCase):
    def setUp(self):
        self.rate = 8000
        self.native = np.full((5*self.rate, 2), .2, dtype=np.float32)
        self.band = np.full_like(self.native, .1)
        self.before = {'text': 'one two three', 'duration': 5,
                       'chunks': [{'text': word, 'timestamp': span} for word, span in
                                  [('one', (.5, .7)), ('two', (2., 2.2)), ('three', (4., 4.2))]]}
        self.after = dict(self.before, text='one wrong three', chunks=[
            {'text': word, 'timestamp': span} for word, span in
            [('one', (.5, .7)), ('wrong', (2., 2.2)), ('three', (4., 4.2))]])
        self.normalize = lambda text: text.lower()

    def propose(self, **kwargs):
        return repair_from_native(self.before, self.before, self.after, self.native, self.band,
                                  self.rate, self.normalize, 'en', **kwargs)

    def test_one_local_stereo_repair_keeps_the_whole_grid_and_inputs(self):
        donor_before, band_before = self.native.copy(), self.band.copy()
        output, metadata = self.propose()
        self.assertEqual(output.shape, self.band.shape)
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_array_equal(self.native, donor_before)
        np.testing.assert_array_equal(self.band, band_before)
        np.testing.assert_array_equal(output[0], self.band[0])
        np.testing.assert_array_equal(output[2*self.rate], self.native[2*self.rate])
        self.assertLess(metadata['repaired_fraction'], .2)
        self.assertFalse(metadata['content_guard_passed'])
        self.assertFalse(metadata['api_checked'])

    def test_donor_errors_inside_the_requested_region_cannot_be_copied(self):
        wrong = dict(self.before, text='one changed three', chunks=[
            {'text': word, 'timestamp': span} for word, span in
            [('one', (.5, .7)), ('changed', (2., 2.2)), ('three', (4., 4.2))]])
        with self.assertRaisesRegex(AlignmentUnavailable, 'Native donor'):
            repair_from_native(self.before, wrong, self.after, self.native, self.band,
                               self.rate, self.normalize, 'en')

    def test_donor_error_elsewhere_does_not_prevent_copying_a_correct_region(self):
        partial = dict(self.before, text='different two three', chunks=[
            {'text': word, 'timestamp': span} for word, span in
            [('different', (.5, .7)), ('two', (2., 2.2)), ('three', (4., 4.2))]])
        output, metadata = repair_from_native(self.before, partial, self.after, self.native, self.band,
                                               self.rate, self.normalize, 'en')
        self.assertFalse(metadata['native_original_unit_parity'])
        self.assertTrue(metadata['donor_matching_regions_only'])
        np.testing.assert_array_equal(output[0], self.band[0])
        np.testing.assert_array_equal(output[2*self.rate], self.native[2*self.rate])

    def test_large_donor_timing_drift_is_rejected(self):
        shifted = dict(self.before, chunks=[
            {'text': word, 'timestamp': span} for word, span in
            [('one', (.5, .7)), ('two', (3., 3.2)), ('three', (4., 4.2))]])
        with self.assertRaises(AlignmentUnavailable):
            repair_from_native(self.before, shifted, self.after, self.native, self.band,
                               self.rate, self.normalize, 'en')

    def test_inserted_candidate_word_uses_its_own_time_interval(self):
        inserted = dict(self.before, text='one extra two three', chunks=[
            {'text': word, 'timestamp': span} for word, span in
            [('one', (.5, .7)), ('extra', (1.2, 1.3)), ('two', (2., 2.2)), ('three', (4., 4.2))]])
        output, metadata = repair_from_native(self.before, self.before, inserted, self.native, self.band,
                                               self.rate, self.normalize, 'en')
        self.assertLess(metadata['repaired_fraction'], .2)
        np.testing.assert_array_equal(output[int(1.25*self.rate)], self.native[int(1.25*self.rate)])
        np.testing.assert_array_equal(output[2*self.rate], self.band[2*self.rate])

    def test_unavailable_timestamp_does_not_produce_a_repair(self):
        missing = dict(self.after, chunks=[{'text': 'one wrong three', 'timestamp': (None, None)}])
        with self.assertRaises(AlignmentUnavailable):
            repair_from_native(self.before, self.before, missing, self.native, self.band,
                               self.rate, self.normalize, 'en')

    def test_widespread_repair_is_rejected_instead_of_copying_the_whole_native_file(self):
        with self.assertRaises(RepairTooLarge):
            self.propose(maximum_coverage=.01)

    def test_unaligned_or_changed_duration_is_rejected(self):
        bad = dict(self.after, duration=4)
        with self.assertRaisesRegex(ValueError, 'durations differ'):
            repair_from_native(self.before, self.before, bad, self.native, self.band,
                               self.rate, self.normalize, 'en')


if __name__ == '__main__':
    unittest.main()
