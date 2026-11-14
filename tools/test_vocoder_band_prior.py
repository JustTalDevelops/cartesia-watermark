"""Check band isolation and exact preservation of the existing unbanded math."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

import vocoder_band_prior as probe


class BandSmoothingTests(unittest.TestCase):
    def setUp(self):
        self.values = np.random.default_rng(9).normal(size=(1, 8, 19)).astype(np.float32)
        self.centers = np.array([0, 1000, 1250, 1500, 3500, 3750, 4000, 4100])
        self.edges = [1000, 1500, 3500, 4000]

    def test_weights_follow_hertz_and_fixed_ramps(self):
        np.testing.assert_array_equal(probe.band_weights(self.centers, self.edges),
                                      [0, 0, .5, 1, 1, .5, 0, 0])

    def test_outside_band_is_exact_gentle_and_plateau_exact_strong(self):
        result = probe.smooth_mel(self.values, 1.5, 1.5, self.centers, self.edges)
        gentle = gaussian_filter(self.values, sigma=(0, .5, .5), mode='reflect')
        strong = gaussian_filter(self.values, sigma=(0, 1.5, 1.5), mode='reflect')
        np.testing.assert_array_equal(result[:, [0, 1, 6, 7]], gentle[:, [0, 1, 6, 7]])
        np.testing.assert_array_equal(result[:, [3, 4]], strong[:, [3, 4]])
        self.assertEqual(result.dtype, np.float32)

    def test_no_band_and_equal_strength_keep_existing_results(self):
        reference = gaussian_filter(self.values, sigma=(0, .5, .5), mode='reflect')
        np.testing.assert_array_equal(probe.smooth_mel(self.values, .5, .5), reference)
        np.testing.assert_array_equal(probe.smooth_mel(self.values, .5, .5, self.centers, self.edges), reference)

    def test_malformed_bands_and_missing_centers_rejected(self):
        for edges in ([1000, 1500, 1500, 4000], [4000, 3500, 1500, 1000], [-1, 1, 2, 3], [0, 1, 2, np.nan]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                probe.band_weights(self.centers, edges)
        with self.assertRaises(ValueError):
            probe.smooth_mel(self.values, 1, 1, None, self.edges)


if __name__ == '__main__':
    unittest.main()
