import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from codec_audio import digest
from experiments.probe_ticodec_band import BANDS,cases_for,native_stem,run


class TiCodecBandTests(unittest.TestCase):
    def test_all_failures_remain_even_when_the_ticodec_stem_failed_quality(self):
        study={'cases':[{'id':name,'marked_provenance':True,'qualified_marked_transition':False,
            'source_sha256':name,'language':'en','detected_language':'en','source_format':{}} for name in ('speech','voice')]}
        ti={'cases':[{'id':name,'source':name,'source_sha256':name,'output_file':name,
            'output_sha256':'ti-'+name,'status':status} for name,status in
            [('speech','content_rejected'),('voice','runtime_quality_rejected')]]}
        frames={'cases':[{'id':name,'source_sha256':name,'treatments':[{'offset_native_samples':0,
            'model':{'candidate_flac_sha256':'native-'+name}}]} for name in ('speech','voice')]}
        cases=cases_for(study,ti,frames)
        self.assertEqual([c['id'] for c in cases],['speech','voice'])
        self.assertEqual(sum(len(c['treatments']) for c in cases),4)
        self.assertTrue(all([t['method'] for t in c['treatments']]==[b[0] for b in BANDS] for c in cases))
        ti['cases'].reverse()
        with self.assertRaisesRegex(ValueError,'original order'):cases_for(study,ti,frames)

    def test_incomplete_stems_stop_before_models_or_output(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);study=root/'study';study.mkdir();ti=root/'ti';ti.mkdir();frame=root/'frame';frame.mkdir()
            for folder in (ti,frame):(folder/'generation.json').write_text('{"complete":false}')
            factory=Mock()
            with self.assertRaisesRegex(ValueError,'Finish both'):
                run(study/'curated.json',ti/'generation.json',frame/'generation.json',root/'new',codec_factory=factory)
            factory.assert_not_called();self.assertFalse((root/'new').exists())

    def test_native_baseline_parity_is_actual_encoded_audio_and_mismatch_is_retained(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.flac'
            t=np.arange(5*8000,dtype=np.float32)/8000
            sf.write(source,.1*np.sin(2*np.pi*331*t),8000,subtype='PCM_24')
            codec=Mock();codec.reconstruct.side_effect=lambda wave,rate,bandwidth_kbps:(wave,1.)
            output=root/'native.flac'
            audio,rate,evidence=native_stem(codec,source,output,digest(source))
            self.assertEqual(evidence['baseline_comparison'],'exact_match');self.assertEqual(rate,8000)
            self.assertEqual(len(audio),40000)
            failed=root/'mismatch.flac'
            with self.assertRaisesRegex(ValueError,'zero-offset'):
                native_stem(codec,source,failed,'mismatched-historical-bytes')
            self.assertTrue(failed.exists())


if __name__=='__main__':unittest.main()
