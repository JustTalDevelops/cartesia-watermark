from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.pcm24_repair import HEADROOM, PCM24_SCALE, blend_pcm24_regions


class CanonicalRepairTests(unittest.TestCase):
    def setUp(self):
        self.base = np.full(40000, 123456/PCM24_SCALE, dtype=np.float32)
        self.base[0] = HEADROOM
        self.donor = np.zeros_like(self.base)
        self.regions = [(2., 2.5)]

    def test_upper_headroom_bin_does_not_rescale_untouched_samples(self):
        output, proof = blend_pcm24_regions(self.base, self.donor, 8000, self.regions)
        np.testing.assert_array_equal(output[:16000], self.base[:16000])
        np.testing.assert_array_equal(output[20000:], self.base[20000:])
        self.assertEqual(output[18000], 0.)
        self.assertEqual(proof['gain'], 1.)
        self.assertTrue(proof['untouched_samples_preserved'])
        self.assertFalse(proof['quality_verified'])

    def test_identical_donor_is_a_bit_exact_zero_operation_in_stereo(self):
        base = np.stack([self.base, -self.base], axis=1)
        output, _ = blend_pcm24_regions(base, base, 8000, self.regions)
        np.testing.assert_array_equal(output, base)

    def test_zero_operation_keeps_the_encoded_pcm24_file_identical(self):
        output, _ = blend_pcm24_regions(self.base, self.base, 8000, self.regions)
        with tempfile.TemporaryDirectory() as temporary:
            original, repaired = Path(temporary)/'original.flac', Path(temporary)/'repaired.flac'
            sf.write(original, self.base, 8000, subtype='PCM_24')
            sf.write(repaired, output, 8000, subtype='PCM_24')
            self.assertEqual(original.read_bytes(), repaired.read_bytes())

    def test_real_headroom_excess_still_receives_a_global_gain(self):
        self.base[0] = np.float32(1-1/PCM24_SCALE)
        output, proof = blend_pcm24_regions(self.base, self.donor, 8000, self.regions)
        self.assertLess(proof['gain'], 1.)
        self.assertFalse(proof['untouched_samples_preserved'])
        self.assertLessEqual(float(np.abs(output).max()), HEADROOM)

    def test_off_grid_nonfinite_and_clipped_inputs_are_rejected(self):
        for value in (.1, np.nan, 1.):
            bad = self.base.copy(); bad[100] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'PCM24'):
                blend_pcm24_regions(bad, self.donor, 8000, self.regions)

    def test_coverage_limit_remains_twenty_percent_and_intervals_stay_bounded(self):
        for regions in ([(0., 2.)], [(-1., 0.)], [(4., 6.)]):
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                blend_pcm24_regions(self.base, self.donor, 8000, regions)
        with self.assertRaises(ValueError):
            blend_pcm24_regions(self.base, self.donor, 8000, self.regions, maximum_coverage=.21)


if __name__ == '__main__':
    unittest.main()
