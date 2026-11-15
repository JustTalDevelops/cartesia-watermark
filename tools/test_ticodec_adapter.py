"""Synthetic-decoder checks; these do not load TiCodec or establish efficacy."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np

from experiments.ticodec_adapter import reconstruct_channels
from experiments.probe_ticodec import run as generate_panel


class TiCodecGridTests(unittest.TestCase):
    def audio(self,rate,channels=1,peak=.1):
        t=np.arange(5*rate+13,dtype=np.float32)/rate
        audio=np.asarray(peak*np.sin(2*np.pi*431*t),dtype=np.float32)
        return audio if channels==1 else np.stack([audio,.4*audio],1)

    def test_complete_mono_and_stereo_grids_at_supported_rates(self):
        for rate in (8000,24000,44100,48000,96000):
            for channels in (1,2):
                with self.subTest(rate=rate,channels=channels):
                    before=self.audio(rate,channels)
                    lengths=[]
                    def identity(wave):lengths.append(len(wave));return wave
                    after,gain=reconstruct_channels(before,rate,identity)
                    self.assertEqual(before.shape,after.shape)
                    self.assertEqual(after.dtype,np.float32)
                    self.assertTrue(np.isfinite(after).all())
                    self.assertTrue(all(n%320==0 for n in lengths))
                    self.assertEqual(len(lengths),channels)
                    self.assertEqual(gain,1.)
                    if channels==2:
                        # Separate float32 resampling accumulates rounding error.
                        np.testing.assert_allclose(after[:,1],.4*after[:,0],atol=np.finfo(np.float32).eps)

    def test_native_identity_discards_only_reflection_without_delay(self):
        before=self.audio(24000)
        after,gain=reconstruct_channels(before,24000,lambda wave:wave)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(gain,1.)

    def test_gain_is_shared_across_complete_channels(self):
        before=self.audio(24000,2,peak=1.4)
        after,gain=reconstruct_channels(before,24000,lambda wave:wave)
        self.assertLess(gain,1.)
        self.assertLessEqual(float(np.abs(after).max()),.9990001)
        np.testing.assert_allclose(after[:,1],.4*after[:,0],atol=6e-8)

    def test_model_length_errors_and_nonfinite_samples_are_not_padded_or_hidden(self):
        before=self.audio(24000)
        for decode in (lambda wave:wave[:-1],lambda wave:wave[None],
                       lambda wave:wave*np.float32(np.nan)):
            with self.assertRaisesRegex(RuntimeError,'finite sample'):
                reconstruct_channels(before,24000,decode)

    def test_invalid_sources_never_reach_a_decoder(self):
        def unexpected(wave):raise AssertionError('decoder must not run')
        for audio,rate in ((np.zeros(4*24000,dtype=np.float32),24000),
                           (np.zeros((5*24000,3),dtype=np.float32),24000),
                           (np.full(5*24000,np.nan,dtype=np.float32),24000)):
            with self.assertRaises(ValueError):reconstruct_channels(audio,rate,unexpected)

    def test_unverified_author_parity_stops_before_models_or_output(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);study=root/'study';study.mkdir()
            report=study/'curated.json';report.write_text('{}')
            parity=root/'parity.json';parity.write_text('{"complete":false}')
            with patch('experiments.probe_ticodec.TiCodec') as model:
                with self.assertRaisesRegex(ValueError,'matching synthetic'):
                    generate_panel(report,parity,root/'development')
            model.assert_not_called()
            self.assertFalse((root/'development').exists())

    def test_previous_development_work_cannot_be_overwritten(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);development=root/'development';development.mkdir()
            with patch('experiments.probe_ticodec.TiCodec') as model:
                with self.assertRaisesRegex(ValueError,'new development directory'):
                    generate_panel(root/'study/curated.json',root/'parity.json',development)
            model.assert_not_called()


if __name__=='__main__':unittest.main()
