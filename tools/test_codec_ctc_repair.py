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

    def test_donor_recognition_must_match_original_before_copying_audio(self):
        wrong = dict(self.before, text='one changed three')
        with self.assertRaisesRegex(AlignmentUnavailable, 'Native donor'):
            repair_from_native(self.before, wrong, self.after, self.native, self.band,
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
