import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('band_delay', ROOT/'experiments/probe_band_group_delay.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class BandGroupDelayTests(unittest.TestCase):
    def test_unit_magnitude_and_requested_group_delay_in_the_plateau(self):
        frequencies = np.arange(0., 8000.125, .125)
        response = probe.phase_response(frequencies, 33.)
        np.testing.assert_allclose(np.abs(response), 1., atol=2e-16)
        observed = -np.diff(np.unwrap(np.angle(response)))/(2*np.pi*np.diff(frequencies))
        midpoint = (frequencies[1:]+frequencies[:-1])/2
        np.testing.assert_allclose(observed[(midpoint>1600)&(midpoint<3400)], .033, atol=1e-12)
        np.testing.assert_allclose(observed[(midpoint<1000)|(midpoint>4000)], 0., atol=1e-12)

    def test_zero_delay_and_stereo_proportions_keep_the_whole_grid(self):
        rate = 16000
        audio = (.1*np.sin(2*np.pi*233*np.arange(rate*3)/rate)).astype(np.float32)
        result, gain = probe.transform(audio, rate, 0.)
        self.assertEqual(result.shape, audio.shape)
        self.assertEqual(gain, 1.)
        np.testing.assert_allclose(result, audio, atol=1e-7)
        stereo, _ = probe.transform(np.column_stack([audio, audio*.5]), rate, 50.)
        self.assertEqual(stereo.shape, (len(audio), 2))
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1]*2)

    def test_steady_tone_amplitude_is_preserved_away_from_recording_edges(self):
        rate = 16000
        audio = (.1*np.sin(2*np.pi*2300*np.arange(rate*5)/rate)).astype(np.float32)
        result, gain = probe.transform(audio, rate, 33.)
        self.assertEqual(gain, 1.)
        interior = result[rate:-rate]
        self.assertAlmostEqual(float(np.sqrt(np.mean(interior**2))), .1/np.sqrt(2), places=4)

    def test_zero_delay_canonical_roundtrip_keeps_bytes_at_a_quantized_peak(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary)/'source.flac', Path(temporary)/'output.flac'
            audio = np.random.default_rng(8).uniform(-.5, .5, 32000).astype(np.float32)
            audio[0], audio[1:20] = np.float32(.9990001), 0.
            probe.sf.write(source, audio, 16000, subtype='PCM_24')
            original, rate = probe.sf.read(source, dtype='float32')
            rebuilt, gain = probe.transform(original, rate, 0.)
            self.assertEqual(gain, 1.)
            canonical = np.rint(rebuilt*np.float32(2**23))/np.float32(2**23)
            probe.sf.write(output, canonical, rate, subtype='PCM_24')
            self.assertEqual(source.read_bytes(), output.read_bytes())

    def test_malformed_delay_grid_and_band_are_rejected(self):
        for value in (-1, 101, np.nan):
            with self.subTest(delay=value), self.assertRaises(ValueError):
                probe.phase_response(np.arange(8001.), value)
        with self.assertRaises(ValueError):
            probe.phase_response(np.arange(8001.)[::-1], 33.)
        with self.assertRaises(ValueError):
            probe.phase_response(np.arange(8001.), 33., [1000, 2000, 1500, 4000])


if __name__ == '__main__':
    unittest.main()
