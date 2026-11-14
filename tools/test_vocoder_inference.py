"""Check context cropping, file safety and format preservation without assets."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from vocoder_inference import synthesize_chunks


class VocoderChunkTests(unittest.TestCase):
    def test_all_frames_are_emitted_once_with_context_removed(self):
        for count in (1, 31, 255, 256, 257, 511, 512, 513, 1031):
            with self.subTest(count=count):
                mel = np.arange(count, dtype=np.float32)[None, None]
                spans = []

                def render(chunk):
                    spans.append(chunk.shape[-1])
                    return np.repeat(chunk.reshape(-1), 3)

                output = synthesize_chunks(mel, 3, render)
                np.testing.assert_array_equal(output, np.repeat(mel.reshape(-1), 3))
                self.assertLessEqual(max(spans), 256+2*32)

    def test_invalid_renderer_outputs_fail(self):
        mel = np.zeros((1, 1, 20), dtype=np.float32)
        for output in (np.zeros(19), np.full(20, np.nan)):
            with self.subTest(size=len(output)), self.assertRaises(RuntimeError):
                synthesize_chunks(mel, 1, lambda _: output)

    def test_invalid_chunk_parameters_fail(self):
        for mel, hop, frames, context in ((np.zeros((1, 0)), 1, 256, 32),
                                          (np.zeros((1, 1)), 0, 256, 32),
                                          (np.zeros((1, 1)), 1, 0, 32),
                                          (np.zeros((1, 1)), 1, 256, -1)):
            with self.assertRaises(ValueError):
                synthesize_chunks(mel, hop, lambda x: x, frames, context)


class VocoderFileTests(unittest.TestCase):
    def test_stereo_length_gain_and_no_overwrite(self):
        import soundfile as sf
        import torch
        from reconstruct_vocoder import PRESET, process_file

        class Model:
            h = SimpleNamespace(sampling_rate=16000, hop_size=16)

            def __call__(self, mel):
                return mel.repeat_interleave(16, dim=-1)*10

        def features(wave, parameters):
            return wave.reshape(1, 1, -1, 16).mean(-1)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, output, metadata = (directory/'in.wav', directory/'out.wav', directory/'out.json')
            x = np.tile(np.array([.2, .1], dtype=np.float32), (4000, 1))
            sf.write(source, x, 8000, subtype='FLOAT')
            original = source.read_bytes()
            result = process_file(Model(), features, source, output, metadata=metadata)
            y, rate = sf.read(output)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(rate, 8000)
            self.assertEqual(result['preset'], PRESET)
            self.assertLessEqual(np.abs(y).max(), .999001)
            np.testing.assert_allclose(y[:, 0], y[:, 1]*2, rtol=1e-6)
            self.assertEqual(source.read_bytes(), original)
            for target, report in ((output, None), (source, None), (directory/'new.wav', source)):
                with self.assertRaises(ValueError):
                    process_file(Model(), features, source, target, metadata=report)
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                process_file(Model(), features, source, directory/'new.wav', preset='unknown')


if __name__ == '__main__':
    unittest.main()
