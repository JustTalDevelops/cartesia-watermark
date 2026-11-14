import json
from pathlib import Path
import tempfile
import unittest

from experiments import codec_study_report as report


class CodecStudyReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.raw, self.output = [self.root/n for n in ('source.flac', 'raw.wav', 'selected.flac')]
        for path in (self.source, self.raw, self.output):
            path.write_bytes(path.name.encode())
        self.source_sha, self.output_sha = report.digest(self.source), report.digest(self.output)
        self.plan, self.freeze = self.root/'plan.json', self.root/'freeze.json'
        self.item = {'id': 'one', 'model': 'fixture', 'language': 'en', 'marked_provenance': True,
                     'verified_output': str(self.output)}
        self.write(self.plan, {'items': [self.item], 'scope_limits': ['fixture']})
        gates = {'turbo_paired_disagreement_maximum': .03, 'turbo_reference_error_delta_maximum': .03,
                 'small_reference_error_delta_maximum': .03, 'mms_reference_error_delta_maximum': .03,
                 'mms_paired_disagreement_maximum': .03, 'three_window_speaker_minimum': .95,
                 'dnsmos_overall_delta_minimum': -.2}
        self.write(self.freeze, {'generation_plan_sha256': report.digest(self.plan), 'source_files_sha256': {},
                                'quality_criteria': gates, 'candidate_implementation_commit': 'fixture',
                                'evaluation_implementation_commit': 'fixture'})
        self.freeze_sha = report.digest(self.freeze)
        self.generation = {'complete': True, 'generation_plan_sha256': report.digest(self.plan),
            'freeze_sha256': self.freeze_sha, 'requests': [{'id': 'one', 'source': str(self.source),
            'sha256': self.source_sha, 'raw_source': str(self.raw), 'raw_sha256': report.digest(self.raw),
            'sample_rate': 44100, 'samples': 441000, 'seconds': 10, 'channels': 1}]}
        self.write(self.root/'generation.json', self.generation)
        generation_sha = report.digest(self.root/'generation.json')
        self.write(self.root/'evaluation.plan.json', {'generation_receipt_sha256': generation_sha})
        common = {'complete': True, 'freeze_sha256': self.freeze_sha, 'generation_receipt_sha256': generation_sha,
                  'http_attempts': [], 'completed_at': '2026-01-01T00:00:01+00:00'}
        check = {'id': 'one', 'sha256': self.source_sha, 'http_status': 200, 'detected': True, 'checked_at': 'fixture'}
        self.baseline = dict(common, cases=[{'id': 'one', 'source_sha256': self.source_sha, 'detected': True}],
                             checks=[dict(check, ordinal=n) for n in (1, 2)])
        attempt = {'preset': 'codec-native', 'state': 'checked', 'sha256': self.output_sha, 'detected': False,
                   'model': {'content_guard_passed': True, 'paired_disagreement': {'turbo': 0, 'mms': 0}}}
        self.candidates = dict(common, started_at='2026-01-01T00:00:02+00:00',
            cases=[{'id': 'one', 'source_sha256': self.source_sha, 'status': 'verified_negative',
                    'output_sha256': self.output_sha, 'negative_checks': 2, 'selected_preset': 'codec-native', 'attempts': [attempt]}],
            checks=[dict(check, role=role, ordinal=n) for role in ('opening_positive_control', 'closing_positive_control') for n in (1, 2)] +
                   [dict(check, role='processed_codec-native', ordinal=n, detected=False, sha256=self.output_sha) for n in (1, 2)])
        self.write(self.root/'baseline.json', self.baseline)
        self.write(self.root/'candidates.json', self.candidates)
        for metric in ('small', 'turbo', 'mms', 'voice'):
            score = {'id': 'one', 'source_sha256': self.source_sha, 'output_sha256': self.output_sha,
                     'status': 'scored', 'paired_disagreement': 0, 'reference_error_delta': 0,
                     'original_reference_error': 0, 'speaker_minimum': .99, 'dnsmos_delta': {'overall': 0}}
            data = {'complete': True, 'originals_only': False, 'plan_sha256': report.digest(self.root/'evaluation.plan.json'),
                    'clips': [score], 'metric': {'model': 'fixture'}}
            self.write(self.root/'scores'/('final_'+metric+'.json'), data)
            if metric != 'voice':
                original = {'id': 'one', 'source_sha256': self.source_sha, 'original_reference_error': 0,
                            'original_reference_error_raw_script': 0, 'original_calibration_eligible': True}
                self.write(self.root/'scores'/('originals_'+metric+'.json'), dict(data, originals_only=True,
                    candidate_outputs_read=False, completed_at=common['completed_at'], clips=[original]))

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def export(self):
        return report.export(self.plan, self.freeze, self.root)

    def test_quality_failure_remains_in_denominator_after_two_negatives(self):
        path = self.root/'scores/final_small.json'
        data = report.load(path)
        data['clips'][0]['reference_error_delta'] = .04
        self.write(path, data)
        result = self.export()
        self.assertEqual(result['independent_requests'], 1)
        self.assertEqual(result['selected_twice_negative'], 1)
        self.assertEqual(result['qualified_checked_outputs'], 0)
        self.assertFalse(result['clips'][0]['quality_gates']['small_reference_error_delta']['passed'])

    def test_insufficient_original_calibration_remains_unproven(self):
        for metric in ('small', 'mms'):
            path = self.root/'scores'/('originals_'+metric+'.json')
            data = report.load(path)
            data['clips'][0]['original_calibration_eligible'] = False
            self.write(path, data)
        result = self.export()
        self.assertEqual(result['all_seven_quality_gates_passed'], 1)
        self.assertEqual(result['qualified_checked_outputs'], 0)
        self.assertEqual(result['clips'][0]['calibrated_checkpoint_count'], 1)

    def test_selected_file_mismatch_is_rejected(self):
        self.output.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'selected file'):
            self.export()

    def test_qualified_transition_does_not_claim_human_approval_or_erasure(self):
        result = self.export()
        self.assertEqual(result['qualified_marked_positive_to_negative'], 1)
        self.assertFalse(result['human_listening_approval'])
        self.assertFalse(result['physical_erasure_proven'])


if __name__ == '__main__':
    unittest.main()
