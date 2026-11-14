import unittest
import numpy as np

from experiments.codec_band_mix import mix_reconstructions


class CodecBandMixTests(unittest.TestCase):
    def setUp(self):
        self.rate = 16000
        self.time = np.arange(5*self.rate)/self.rate

    def test_tones_select_codec_in_band_and_vocoder_outside(self):
        low = np.sin(2*np.pi*300*self.time)
        high = np.sin(2*np.pi*2500*self.time)
        vocoder = (.1*low-.05*high).astype(np.float32)
        codec = (.02*low+.1*high).astype(np.float32)
        original = vocoder.copy()
        output, gain = mix_reconstructions(vocoder, codec, self.rate)
        segment = slice(self.rate, -self.rate)
        self.assertEqual(gain, 1.)
        self.assertEqual(output.shape, vocoder.shape)
        self.assertAlmostEqual(2*np.mean(output[segment]*low[segment]), .1, places=3)
        self.assertAlmostEqual(2*np.mean(output[segment]*high[segment]), .1, places=3)
        np.testing.assert_array_equal(vocoder, original)

    def test_stereo_channels_keep_their_relative_levels(self):
        tone = (.1*np.sin(2*np.pi*2500*self.time)).astype(np.float32)
        codec = np.column_stack([tone, tone*.5])
        output, gain = mix_reconstructions(np.zeros_like(codec), codec, self.rate)
        self.assertEqual(output.shape, codec.shape)
        self.assertEqual(gain, 1.)
        np.testing.assert_array_equal(output[:, 0], output[:, 1]*2)

    def test_rejects_mismatched_samples_nonfinite_and_invalid_band(self):
        audio = np.zeros(len(self.time), dtype=np.float32)
        with self.assertRaises(ValueError):
            mix_reconstructions(audio[:-1], audio, self.rate)
        audio[0] = np.nan
        with self.assertRaises(ValueError):
            mix_reconstructions(audio, audio, self.rate)
        audio[0] = 0
        with self.assertRaises(ValueError):
            mix_reconstructions(audio, audio, self.rate, high=self.rate/2)

    def test_shared_gain_limits_peaks_without_normalizing_channels_separately(self):
        audio = np.full((len(self.time), 2), [1.2, .6], dtype=np.float32)
        output, gain = mix_reconstructions(audio, audio, self.rate)
        self.assertLess(gain, 1.)
        self.assertLessEqual(float(np.abs(output).max()), 1.)
        np.testing.assert_array_equal(output[:, 0], output[:, 1]*2)


if __name__ == '__main__':
    unittest.main()
