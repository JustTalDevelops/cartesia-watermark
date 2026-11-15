import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests
import soundfile as sf

from codec_audio import digest
from experiments.probe_codec_frames import (PHASES, ROOT, failed_marked_cases, generate_case,
    historical_native_sha, run, without_http)


class FixtureRuntime:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.calls = []
        self.codec = None

    def clear_job(self):
        pass

    def process_file(self, source, output, preset, metadata):
        self.calls.append(self.codec.offset_samples)
        if self.accepted:
            audio, rate = sf.read(source, dtype='float32')
            sf.write(output, audio, rate, subtype='FLOAT')
        return {'content_guard_passed':self.accepted, 'candidate_flac_sha256':digest(source)}


class FrameProbeTests(unittest.TestCase):
    def test_incomplete_study_never_creates_output_or_loads_a_codec(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); study = root/'study';study.mkdir()
            report = study/'curated_complete_report.json';report.write_text('{}')
            for name in PHASES:
                (study/(name+'.json')).write_text(json.dumps({'complete':name!='candidates'}))
            factory = Mock()
            with self.assertRaisesRegex(ValueError, 'Finish the current study'):
                run(report, root/'probe', codec_factory=factory)
            factory.assert_not_called()
            self.assertFalse((root/'probe').exists())

    def test_complete_flags_without_matching_curated_observations_do_not_load_models(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary);study = root/'study';study.mkdir()
            report = study/'curated_complete_report.json'
            report.write_text(json.dumps({'kind':'Frozen prospective language-profile study',
                'planned_requests':48,'marked_provenance_requests':44,
                'cases':[{'id':str(i)} for i in range(48)],
                'freeze_sha256':digest(ROOT/'results/codec_profiles_reliability.freeze.json')}))
            for name in PHASES:(study/(name+'.json')).write_text('{"complete":true}')
            factory = Mock()
            with patch('experiments.codec_profiles_report.export', return_value={'different':'observations'}):
                with self.assertRaisesRegex(ValueError, 'actual observations'):
                    run(report, root/'probe', codec_factory=factory)
            factory.assert_not_called()
            self.assertFalse((root/'probe').exists())

    def test_every_failed_marked_case_remains_and_nominal_controls_are_separate(self):
        rows = [{'id':'speech','marked_provenance':True,'qualified_marked_transition':False},
                {'id':'later-quality','marked_provenance':True,'qualified_marked_transition':False},
                {'id':'qualified','marked_provenance':True,'qualified_marked_transition':True},
                {'id':'nominal','marked_provenance':False,'qualified_marked_transition':False}]
        self.assertEqual([row['id'] for row in failed_marked_cases({'cases':rows})], ['speech','later-quality'])
        self.assertEqual(historical_native_sha({'attempts':[{'model':{'native_codec_sha256':'one'}}]*2}), 'one')
        with self.assertRaisesRegex(ValueError, 'baselines disagree'):
            historical_native_sha({'attempts':[{'model':{'native_codec_sha256':value}} for value in ('one','two')]})

    def fixture(self, root, expected=None):
        source = root/'source.flac'
        t = np.arange(5*8000, dtype=np.float32)/8000
        sf.write(source, .1*np.sin(2*np.pi*231*t), 8000, subtype='PCM_24')
        row = {'id':'fixture','source_sha256':digest(source),'historical_native_sha256':expected,
               'treatments':[{'offset_native_samples':offset,'status':'planned'} for offset in (0,80,160,240)]}
        base = Mock(metadata={'model':'OFFLINE_FIXTURE'})
        return source,row,base

    def test_baseline_mismatch_stops_before_another_offset_or_voice_estimate(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source,row,base = self.fixture(root, 'different')
            runtime = FixtureRuntime(); voice = Mock()
            with self.assertRaisesRegex(ValueError, 'historical bytes'):
                generate_case(runtime,base,source,row,root/'case',voice,lambda:None)
            self.assertEqual(runtime.calls,[0])
            voice.assert_not_called()
            self.assertEqual(row['treatments'][0]['status'],'baseline_mismatch')
            self.assertTrue(all(t['status']=='planned' for t in row['treatments'][1:]))

    def test_speech_rejections_keep_all_treatments_without_voice_or_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary);source,row,base = self.fixture(root)
            runtime = FixtureRuntime(accepted=False);voice = Mock()
            generate_case(runtime,base,source,row,root/'case',voice,lambda:None)
            self.assertEqual(runtime.calls,[0,80,160,240])
            self.assertTrue(all(t['status']=='content_rejected' for t in row['treatments']))
            self.assertFalse(list((root/'case').glob('*.flac')))
            voice.assert_not_called()

    def test_voice_rejections_are_retained_for_every_exact_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary);source,row,base = self.fixture(root)
            row['historical_native_sha256'] = digest(source)
            runtime = FixtureRuntime();voice = Mock(return_value={'passed':False})
            generate_case(runtime,base,source,row,root/'case',voice,lambda:None)
            self.assertEqual(voice.call_count,4)
            self.assertTrue(all(t['status']=='runtime_quality_rejected' for t in row['treatments']))
            self.assertEqual(row['treatments'][0]['historical_baseline_comparison'],'exact_match')
            self.assertTrue(all(digest(Path(t['output_file']))==digest(source) for t in row['treatments']))
            self.assertFalse(list((root/'case').glob('*.wav')))

    def test_http_block_restores_the_original_transport_without_sending_a_request(self):
        previous = requests.sessions.Session.request
        with without_http(),self.assertRaisesRegex(RuntimeError, 'HTTP requests are disabled'):
            requests.get('https://example.invalid')
        self.assertIs(requests.sessions.Session.request, previous)


if __name__=='__main__':
    unittest.main()
