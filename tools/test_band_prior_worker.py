import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from codec_audio import digest
from reconstruct_codec_band import BandRuntime


class BandPriorWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.native = self.root/'source.wav', self.root/'native.wav'
        self.output = self.root/'result.wav'
        self.audio = (.1*np.sin(2*np.pi*231*np.arange(40000)/8000)).astype(np.float32)
        sf.write(self.source, self.audio, 8000, subtype='FLOAT')
        sf.write(self.native, np.zeros_like(self.audio), 8000, subtype='FLOAT')
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'source': str(self.source), 'source_sha256': digest(self.source),
                                             'native': str(self.native), 'native_sha256': digest(self.native)}))
        self.runtime = BandRuntime.__new__(BandRuntime)
        self.runtime.model, self.runtime.features = object(), object()
        self.runtime.assets = {'model_id': 'fixture', 'model_revision': 'fixture',
                               'model_sha256': {'bigvgan_generator.pt': 'fixture'}}

    def render(self, model, features, audio, rate, frequency_sigma, time_sigma, edges):
        self.assertEqual((rate, frequency_sigma, time_sigma), (8000, 1., 1.))
        self.assertEqual(tuple(edges), (1000., 1500., 3500., 4000.))
        return audio*.93, 1.

    def test_prior_uses_the_original_condition_without_native_waveform_mix(self):
        with patch('vocoder_band_prior.reconstruct', side_effect=self.render), \
                patch.object(self.runtime, 'reconstruct_base', side_effect=AssertionError('old base called')), \
                patch('reconstruct_codec_band.mix_reconstructions', side_effect=AssertionError('native mix called')):
            report = self.runtime.process_file(self.manifest, self.output, 'codec-band-prior')
        actual, rate = sf.read(self.output, dtype='float32')
        self.assertEqual((rate, len(actual)), (8000, len(self.audio)))
        np.testing.assert_allclose(actual, self.audio*.93, atol=2**-23)
        self.assertFalse(report['native_input_used_for_synthesis'])
        self.assertFalse(report['quality_verified'])
        self.assertEqual(report['output_sha256'], digest(self.output))
        np.testing.assert_array_equal(actual*2**23, np.rint(actual*2**23))

    def test_changed_source_stops_before_publication(self):
        def change_source(*args):
            result = self.render(*args)
            self.source.write_bytes(b'changed')
            return result
        with patch('vocoder_band_prior.reconstruct', side_effect=change_source):
            with self.assertRaisesRegex(RuntimeError, 'changed during synthesis'):
                self.runtime.process_file(self.manifest, self.output, 'codec-band-prior')
        self.assertFalse(self.output.exists())

    def test_prior_accepts_a_source_only_manifest_but_legacy_band_requires_a_donor(self):
        self.manifest.write_text(json.dumps({'source': str(self.source), 'source_sha256': digest(self.source)}))
        with patch('vocoder_band_prior.reconstruct', side_effect=self.render):
            report = self.runtime.process_file(self.manifest, self.output, 'codec-band-prior')
        self.assertIsNone(report['native_sha256'])
        with self.assertRaisesRegex(ValueError, 'requires its native donor'):
            self.runtime.process_file(self.manifest, self.root/'other.wav', 'codec-band')


if __name__ == '__main__':
    unittest.main()
