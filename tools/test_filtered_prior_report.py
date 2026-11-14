"""Private regression summaries must not disclose individual recordings."""
import json
import unittest

from experiments.filtered_prior_report import regression_summary


class PrivateRegressionSummaryTests(unittest.TestCase):
    def test_only_aggregate_metrics_leave_private_rows(self):
        row = {'id': 'private-recording-id', 'source_sha256': 'private-source-fingerprint',
               'output_sha256': 'private-output-fingerprint', 'source': '/private/recording.wav',
               'content': {'unit': 'word', 'asr_disagreement': 0., 'transcript': 'Private spoken words'},
               'secondary_content': {'unit': 'word', 'asr_disagreement': 0.},
               'voice': {'speaker_minimum': .99, 'dnsmos_delta': {'overall': .1},
                         'embeddings': ['private-biometric-vector']}}
        key = row['id']
        result = regression_summary([row], [row], {key: {'detected': True}},
                                    {key: {'detected': False}}, {key: {'detected': False}},
                                    {key: {'status': 'verified_negative', 'selected_preset': 'filtered-prior'}}, 1)
        serialized = json.dumps(result)
        for secret in (key, row['source_sha256'], row['output_sha256'], row['source'],
                       'Private spoken words', 'private-biometric-vector'):
            self.assertNotIn(secret, serialized)
        self.assertEqual(result['selected_verified_negative'], 1)
        self.assertFalse(result['selected_quality']['known_text_criteria_applicable'])
        self.assertNotIn('recordings_meeting_all_estimated_criteria', result['selected_quality'])


if __name__ == '__main__':
    unittest.main()
