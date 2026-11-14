from copy import deepcopy
import unittest

import numpy as np
import torch

from codec_inference import NativeCodec
from experiments.codec_frame_alignment import FrameAlignedCodec, DEVELOPMENT_OFFSETS


class IdentityDecoder:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, bandwidth_id):
        self.calls.append((audio.clone(), bandwidth_id.item(), torch.is_inference_mode_enabled()))
        return audio.clone()


class FrameMeanDecoder:
    def __call__(self, audio, bandwidth_id):
        return audio.reshape(1, -1, 320).mean(-1).repeat_interleave(320, dim=-1)


class FrameAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def codec(self, model=None):
        # No real model or recognizer is constructed by these offline tests.
        codec = object.__new__(NativeCodec)
        codec.model = model if model is not None else IdentityDecoder()
        codec.metadata = {'model': 'TEST_FIXTURE', 'precision': 'float32', 'nested': {'value': 1}}
        return codec

    def audio(self, rate=24000, channels=1):
        t = np.arange(5*rate+137, dtype=np.float32)/rate
        first = (.12*np.sin(2*np.pi*231*t)).astype(np.float32)
        return first if channels == 1 else np.stack((first, .07*np.cos(2*np.pi*417*t)), axis=1).astype(np.float32)

    def test_zero_offset_delegates_to_the_existing_codec_and_shares_weights(self):
        base = self.codec()
        metadata = deepcopy(base.metadata)
        source = self.audio()
        wrapper = FrameAlignedCodec(base, 0)
        expected = base.reconstruct(source, 24000, bandwidth_kbps=6.)
        actual = wrapper.reconstruct(source, 24000, bandwidth_kbps=6.)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertIs(wrapper.codec.model, base.model)
        self.assertEqual(base.metadata, metadata)
        self.assertEqual(base.model.calls[-1][1], 2)
        self.assertEqual(len(wrapper.metadata['frame_alignment_code_sha256']), 64)
        with self.assertRaises(AttributeError):wrapper.offset_samples = 80

    def test_offsets_are_undone_before_resampling_and_preserve_complete_channels(self):
        for rate, channels in ((8000, 1), (24000, 2), (44100, 1), (48000, 2), (96000, 1)):
            source = self.audio(rate, channels)
            base = self.codec()
            expected, expected_gain = base.reconstruct(source, rate)
            for offset in DEVELOPMENT_OFFSETS[1:]:
                with self.subTest(rate=rate, channels=channels, offset=offset):
                    actual, gain = FrameAlignedCodec(base, offset).reconstruct(source, rate)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.shape, source.shape)
                    self.assertEqual(gain, expected_gain)
                    self.assertTrue(all(call[2] for call in base.model.calls))

    def test_offset_reaches_the_lossy_frame_decoder_instead_of_just_shifting_output(self):
        source = self.audio()
        base = self.codec(FrameMeanDecoder())
        first, _ = FrameAlignedCodec(base, 0).reconstruct(source, 24000)
        second, _ = FrameAlignedCodec(base, 80).reconstruct(source, 24000)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(first.shape, second.shape)
        self.assertTrue(np.isfinite(second).all())

    def test_wrong_length_and_nonfinite_decoder_results_are_rejected(self):
        for bad in ('length', 'nan'):
            def decode(audio, bandwidth_id):
                return audio[:, :-1] if bad == 'length' else torch.full_like(audio, float('nan'))
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                FrameAlignedCodec(self.codec(decode), 80).reconstruct(self.audio(), 24000)

    def test_shared_gain_limits_both_channels_without_clipping(self):
        def amplify(audio, bandwidth_id):
            return audio*20
        source = self.audio(channels=2)
        result, gain = FrameAlignedCodec(self.codec(amplify), 160).reconstruct(source, 24000)
        self.assertLess(gain, 1.)
        self.assertAlmostEqual(float(np.max(np.abs(result))), .999, places=6)
        np.testing.assert_allclose(result, source*(20*gain), atol=1e-7)

    def test_invalid_offsets_bandwidths_and_audio_fail_before_decoding(self):
        base = self.codec()
        for offset in (-1, 320, .5, True):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                FrameAlignedCodec(base, offset)
        for bandwidth in (0, 5., 24.):
            with self.subTest(bandwidth=bandwidth), self.assertRaises(ValueError):
                FrameAlignedCodec(base, 80).reconstruct(self.audio(), 24000, bandwidth)
        invalid = self.audio(); invalid[0] = np.nan
        with self.assertRaises(ValueError):FrameAlignedCodec(base, 80).reconstruct(invalid, 24000)
        self.assertFalse(base.model.calls)


if __name__ == '__main__':
    unittest.main()
