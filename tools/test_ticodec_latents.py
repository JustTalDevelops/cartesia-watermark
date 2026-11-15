import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import numpy as np

from experiments.ticodec_latents import MODES,reconstruct_modes
from experiments.probe_ticodec_latents import planned_cases,run


class LatentGridTests(unittest.TestCase):
    def test_encoder_is_shared_once_per_complete_channel_and_control_has_no_native_delay(self):
        t=np.arange(120013,dtype=np.float32)/24000
        audio=np.stack([.1*np.sin(2*np.pi*371*t),.04*np.sin(2*np.pi*371*t)],1)
        encoded=[];decoded=[]
        def encode(wave):encoded.append(len(wave));return wave
        def decode(state,mode):decoded.append(mode);return state
        result=reconstruct_modes(audio,24000,encode,decode)
        self.assertEqual(len(encoded),2)
        self.assertEqual(decoded,[mode for mode,_,_ in MODES for channel in range(2)])
        for wave,gain in result.values():
            np.testing.assert_array_equal(wave,audio);self.assertEqual(gain,1.)

    def test_control_mismatch_stops_before_any_other_decoder_cell(self):
        audio=np.zeros(120000,dtype=np.float32);decoded=[]
        def decode(state,mode):decoded.append(mode);return state
        def mismatch(wave,gain):raise ValueError('baseline mismatch')
        with self.assertRaisesRegex(ValueError,'baseline mismatch'):
            reconstruct_modes(audio,24000,lambda wave:wave,decode,mismatch)
        self.assertEqual(decoded,['quantized-control'])

    def test_all_cells_keep_mono_and_stereo_grids_across_supported_rates(self):
        for rate in (8000,24000,44100,48000,96000):
            for channels in (1,2):
                with self.subTest(rate=rate,channels=channels):
                    t=np.arange(5*rate+13,dtype=np.float32)/rate
                    audio=np.asarray(.1*np.sin(2*np.pi*431*t),dtype=np.float32)
                    if channels==2:audio=np.stack([audio,.4*audio],1)
                    result=reconstruct_modes(audio,rate,lambda wave:wave,lambda state,mode:state)
                    self.assertEqual(list(result),[name for name,_,_ in MODES])
                    for wave,gain in result.values():
                        self.assertEqual(wave.shape,audio.shape);self.assertEqual(wave.dtype,np.float32)
                        self.assertTrue(np.isfinite(wave).all());self.assertEqual(gain,1.)

    def test_decoder_length_errors_cannot_be_hidden_with_padding(self):
        with self.assertRaisesRegex(RuntimeError,'malformed'):
            reconstruct_modes(np.zeros(120000,dtype=np.float32),24000,lambda wave:wave,
                              lambda state,mode:state[:-1])

    def test_all_failed_sources_and_three_new_cells_remain(self):
        study={'cases':[{'id':name,'marked_provenance':True,'qualified_marked_transition':False,
            'source_sha256':name,'language':'en','detected_language':'en'} for name in ('speech','voice')]}
        previous={'cases':[{'id':name,'source':name,'output_sha256':'old-'+name,'status':status}
            for name,status in [('speech','content_rejected'),('voice','runtime_quality_rejected')]]}
        cases=planned_cases(study,previous)
        self.assertEqual([c['id'] for c in cases],['speech','voice'])
        self.assertEqual(sum(len(c['treatments']) for c in cases),6)
        self.assertTrue(all(len(c['treatments'])==3 for c in cases))

    def test_incomplete_original_panel_stops_before_new_models_or_output(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);old=root/'old';old.mkdir();study=root/'study';study.mkdir()
            previous=old/'generation.json';previous.write_text('{"complete":false}')
            factory=Mock()
            with self.assertRaisesRegex(ValueError,'Finish the original'):
                run(study/'curated.json',previous,root/'new',model_factory=factory)
            factory.assert_not_called();self.assertFalse((root/'new').exists())


if __name__=='__main__':unittest.main()
