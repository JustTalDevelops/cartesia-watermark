from collections import deque
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

from tools import test_codec_frame_endpoint as fixtures
from experiments import check_codec_frames as common
from experiments import check_ticodec_panel as panel


class TiCodecEndpointTests(unittest.TestCase):
    def fixture(self,root):
        value,declaration,output=fixtures.FrameEndpointTests().fixture(root)
        for row in value['treatments'][1:]:
            row.pop('offset_native_samples');row['method']='band-wide'
        value['treatments'][2]['method']='band-narrow'
        value['treatments'][0].update(processing_status='processing_eligible',
            endpoint_phase_status='qualified_pilot_reused',prior_candidate_detected=False)
        value['observation_report_kind']='Joint method-labelled offline fixture'
        declaration.write_text(json.dumps(dict(value,declared_at=common.now())))
        return value,declaration,output

    def execute(self,value,declaration,output,builder):
        replies=deque([True,True,False,False,True,True])
        post=Mock(side_effect=lambda *a,**k:Mock(status_code=200,json=lambda:{'detected':replies.popleft()}))
        with patch.object(common.cartesia.requests,'post',post), \
                patch.object(common.cartesia,'ROOT',declaration.parent), \
                patch.object(common.cartesia,'_pace_detector',return_value=None):
            common.run(declaration,output,builder=builder)
        return post

    def test_named_methods_and_prior_pilot_remain_without_fake_offsets_or_extra_calls(self):
        with TemporaryDirectory() as temporary:
            value,declaration,output=self.fixture(temporary)
            post=self.execute(value,declaration,output,lambda *args:value)
            report=json.loads(output.read_text())
            self.assertTrue(report['complete']);self.assertEqual(post.call_count,6)
            self.assertEqual(report['kind'],value['observation_report_kind'])
            self.assertEqual(report['treatments'][0]['endpoint_phase_status'],'qualified_pilot_reused')
            candidates=[r for r in report['checks'] if r['role']=='candidate']
            self.assertEqual([r['method'] for r in candidates],['band-wide','band-wide'])
            self.assertTrue(all('offset_native_samples' not in r for r in candidates))
            self.assertTrue(report['treatments'][2]['identical_file_observations_reused'])

    def test_changed_inventory_at_completion_cannot_be_qualified(self):
        with TemporaryDirectory() as temporary:
            value,declaration,output=self.fixture(temporary)
            builder=Mock(side_effect=[value,dict(value,maximum_http_attempts=99)])
            with self.assertRaisesRegex(ValueError,'inventory changed'):
                self.execute(value,declaration,output,builder)
            report=json.loads(output.read_text())
            self.assertFalse(report['complete']);self.assertEqual(len(report['http_attempts']),6)

    def test_incomplete_panels_stop_before_any_quality_export_or_endpoint(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ['codec_profiles_fresh','ticodec_development','codec_frame_development','ticodec_band_development']:
                (root/name).mkdir()
            study=root/'codec_profiles_fresh/curated_complete_report.json';study.write_text('{}')
            for name in ['ticodec_development','codec_frame_development']:
                (root/name/'generation.json').write_text('{"complete":true}')
            band=root/'ticodec_band_development/generation.json';band.write_text('{"complete":false}')
            with patch.object(panel,'generation_export') as exporter,patch.object(panel.endpoint.cartesia.requests,'post') as post:
                with self.assertRaisesRegex(ValueError,'Finish and preserve'):panel.data(band,study,root/'code')
            exporter.assert_not_called();post.assert_not_called()


if __name__=='__main__':unittest.main()
