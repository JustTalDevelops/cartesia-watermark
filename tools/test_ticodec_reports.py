"""Malformed receipt/inventory checks; no audio, models or API requests."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from codec_audio import digest
from experiments.ticodec_pilot_report import export as pilot_report
from experiments.ticodec_generation_report import export as generation_report


class TiCodecReportingTests(unittest.TestCase):
    def pilot_fixture(self,root):
        pilot=root/'ticodec_pilot';pilot.mkdir()
        ti=root/'ticodec_development';ti.mkdir()
        declaration={'source_sha256':'source','candidate_sha256':'candidate',
            'assessment_sha256':'assessment','helper_source_sha256':'helper',
            'declared_at':'2026-01-01T00:00:00+00:00'}
        (root/'ticodec_pilot.declaration.json').write_text(json.dumps(declaration))
        (ti/'generation.json').write_text('{"complete":true}')
        checks=[];attempts=[];start=datetime(2026,1,1,tzinfo=timezone.utc)
        for index,(role,ordinal) in enumerate([('opening_original',1),('opening_original',2),
                ('candidate',1),('candidate',2),('closing_original',1),('closing_original',2)]):
            row={'role':role,'ordinal':ordinal,'sha256':'candidate' if role=='candidate' else 'source','http_status':200}
            checks.append(dict(row,detected=role!='candidate'))
            attempts.append(dict(row,started_at=(start+timedelta(seconds=1+80*index)).isoformat()))
        report={'complete':True,'candidate_detected':False,'source_sha256':'source','candidate_sha256':'candidate',
            'assessment_sha256':'assessment','checker_source_sha256':'helper','checks':checks,'http_attempts':attempts}
        return pilot/'result.json',report

    def test_missing_actual_check_cannot_be_qualified(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);path,report=self.pilot_fixture(root)
            report['checks'].pop();path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'six-request'):pilot_report(root)

    def test_failed_positive_control_stops_before_audio_or_reference_reads(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);path,report=self.pilot_fixture(root)
            report['checks'][-1]['detected']=False;path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'receipts conflict'):pilot_report(root)

    def test_generation_report_never_drops_a_failed_source(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);F=root/'codec_profiles_fresh';F.mkdir();T=root/'ticodec_development';T.mkdir()
            study=F/'curated_complete_report.json'
            study.write_text(json.dumps({'cases':[{'id':'failed','marked_provenance':True,'qualified_marked_transition':False}]}))
            (T/'generation.json').write_text(json.dumps({'complete':True,'completed_study_sha256':digest(study),
                'cases':[],'planned_spent_sources':0,'planned_treatments':0}))
            with self.assertRaisesRegex(ValueError,'every spent marked failure'):generation_report(root)


if __name__=='__main__':unittest.main()
