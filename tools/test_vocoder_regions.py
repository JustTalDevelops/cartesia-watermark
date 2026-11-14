"""Sparse synthesis must retain global block boundaries and model context."""
import unittest

import numpy as np

from vocoder_inference import synthesize_chunks
from vocoder_regions import synthesize_regions


class VocoderRegionTests(unittest.TestCase):
    def test_used_samples_match_full_context_sensitive_render(self):
        mel = np.arange(1031, dtype=np.float32)[None, None]
        calls = []

        def render(chunk):
            calls.append(chunk.shape[-1])
            # Deliberately depends on both context boundaries, so changing the
            # global block or context schedule changes every emitted sample.
            return np.repeat(chunk.reshape(-1)+chunk.sum(), 3)

        full = synthesize_chunks(mel, 3, render)
        calls.clear()
        regions = [(775, 785), (3090, 3093), (780, 790)]
        sparse, blocks = synthesize_regions(mel, 3, render, regions)
        self.assertEqual(blocks, (1, 4))
        self.assertEqual(calls, [320, 39])
        for start, end in regions:
            np.testing.assert_array_equal(sparse[start:end], full[start:end])
        self.assertFalse(sparse[:768].any())
        self.assertFalse(sparse[1536:3072].any())

    def test_all_blocks_equal_full_waveform_including_short_tail(self):
        for count in (1, 255, 256, 257, 512, 513):
            mel = np.arange(count, dtype=np.float64)[None, None]
            render = lambda chunk: np.repeat(chunk.reshape(-1), 2)
            result, blocks = synthesize_regions(mel, 2, render, [(0, count*2)])
            np.testing.assert_array_equal(result, synthesize_chunks(mel, 2, render))
            self.assertEqual(blocks, tuple(range((count+255)//256)))

    def test_invalid_intervals_never_call_renderer(self):
        mel = np.ones((1, 1, 10), dtype=np.float32)
        for regions in ([], [(0, 0)], [(-1, 1)], [(0, 11)], [(0., 2)], [(False, 2)], [(1,)], [(2, 1)]):
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                synthesize_regions(mel, 1, lambda _: self.fail('Invalid interval rendered'), regions)

    def test_invalid_model_results_are_rejected(self):
        mel = np.ones((1, 1, 10), dtype=np.float32)
        for generated in (np.zeros(9), np.full(10, np.nan), np.ones(10, dtype=np.int32)):
            with self.subTest(dtype=generated.dtype), self.assertRaises(RuntimeError):
                synthesize_regions(mel, 1, lambda _: generated, [(1, 2)])


class RegionalReconstructionTests(unittest.TestCase):
    def test_used_samples_match_full_unscaled_render_across_rates_and_channels(self):
        from types import SimpleNamespace
        import torch
        from reconstruct_vocoder import reconstruct
        from reconstruct_vocoder_regions import reconstruct_regions

        class Model:
            h = SimpleNamespace(sampling_rate=44100, hop_size=16)

            def __call__(self, mel):
                return mel.repeat_interleave(16, dim=-1)

        def features(wave, parameters):
            return wave.reshape(1, 1, -1, 16).mean(-1)

        intervals = [(.01, .025), (.181, .19), (.69, .705)]
        torch.set_num_threads(2)
        for rate in (8000, 16000, 44100, 48000, 96000):
            clock = np.arange(round(rate*.71), dtype=np.float32)/rate
            mono = (.15*np.sin(clock*440*2*np.pi)+.07*np.sin(clock*1200*2*np.pi)).astype(np.float32)
            for audio in (mono, np.stack((mono, mono*.3), axis=1)):
                with self.subTest(rate=rate, channels=audio.ndim):
                    full, gain = reconstruct(Model(), features, audio, rate)
                    self.assertEqual(gain, 1)
                    sparse, report = reconstruct_regions(Model(), features, audio, rate, intervals)
                    self.assertEqual(sparse.shape, audio.shape)
                    self.assertTrue(report['waveform_is_partial'])
                    self.assertTrue(report['donor_is_unscaled'])
                    self.assertTrue(all(len(b) < report['full_blocks_per_channel']
                                        for b in report['rendered_blocks_per_channel']))
                    for start, end in intervals:
                        np.testing.assert_array_equal(sparse[int(start*rate):int(np.ceil(end*rate))],
                                                      full[int(start*rate):int(np.ceil(end*rate))])


if __name__ == '__main__':
    unittest.main()
