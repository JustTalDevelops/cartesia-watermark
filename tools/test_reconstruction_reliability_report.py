"""Prevent denominator inflation and duplicate measurement accounting."""
import unittest

from experiments.reconstruction_reliability_report import (content_metrics, indexed, paired_counts,
                                                            selected_quality_rows, summarize_content,
                                                            verification_summary)


class ReliabilityAccountingTests(unittest.TestCase):
    def test_original_negatives_do_not_inflate_removal_successes(self):
        rows = [{'original_detected': before, 'processed_detected': after}
                for before, after in ((True, False), (True, True), (False, False), (False, True))]
        result = paired_counts(rows)
        self.assertEqual(result['recordings'], 4)
        self.assertEqual(result['original_positive'], 2)
        self.assertEqual(result['positive_to_negative'], 1)
        self.assertEqual(result['remaining_positive'], 1)
        self.assertEqual(result['original_negative'], 2)
        self.assertEqual(result['negative_to_positive'], 1)
        self.assertEqual(result['negative_to_negative'], 1)

    def test_duplicate_measurements_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            indexed([{'id': 'a'}, {'id': 'a'}], {'a'}, 'API')

    def test_missing_and_unexpected_measurements_are_rejected(self):
        for rows in ([{'id': 'a'}], [{'id': 'a'}, {'id': 'b'}, {'id': 'extra'}]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'Incomplete or unexpected'):
                indexed(rows, {'a', 'b'}, 'API')

    def test_raw_transcripts_and_private_paths_are_not_exported(self):
        metrics = content_metrics({'unit': 'word', 'asr_disagreement': .1,
                                   'transcript': 'Private recording contents', 'source': '/private/recording.wav'})
        self.assertEqual(metrics, {'unit': 'word', 'asr_disagreement': .1})

    def test_verification_summary_excludes_private_audio_paths(self):
        audit = {'status': 'verified_negative', 'quality_verified': False, 'boundary_safe_requested': True,
                 'source': '/private/input.wav', 'output': '/private/output.wav',
                 'source_sha256': 'input-hash', 'output_sha256': 'output-hash',
                 'selected_preset': 'full-flow', 'checked_at': 'timestamp',
                 'attempts': [{'sha256': 'output-hash', 'http_status': 200, 'detected': False}]}
        import json
        summary = verification_summary(audit, {'torch_installed': False})
        self.assertNotIn('/private/', json.dumps(summary))
        self.assertEqual(summary['output_sha256'], 'output-hash')
        audit['attempts'][0]['detected'] = True
        with self.assertRaises(ValueError):
            verification_summary(audit, {})

    def test_summary_requires_evidence_for_each_requested_confirmation(self):
        audit = {'status': 'verified_negative', 'quality_verified': False, 'boundary_safe_requested': True,
                 'source_sha256': 'input', 'output_sha256': 'output', 'selected_preset': 'full-flow',
                 'checked_at': 'timestamp', 'required_negative_checks': 2,
                 'attempts': [{'sha256': 'output', 'http_status': 200, 'detected': False}]}
        with self.assertRaisesRegex(ValueError, 'required negative confirmations'):
            verification_summary(audit, {})
        checks = [{'http_status': 200, 'detected': False} for _ in range(2)]
        audit['attempts'][0]['checks'] = checks
        self.assertEqual(verification_summary(audit, {})['observed_negative_checks'], 2)
        checks[0]['detected'] = True
        with self.assertRaisesRegex(ValueError, 'required negative confirmations'):
            verification_summary(audit, {})

    def test_character_and_word_errors_have_separate_summaries(self):
        summary = summarize_content([
            {'unit': 'word', 'asr_disagreement': 0., 'reference_error_delta': .01},
            {'unit': 'word', 'asr_disagreement': .2},
            {'unit': 'character', 'asr_disagreement': .6, 'reference_error_delta': .3},
        ])
        self.assertEqual(summary['word']['recordings'], 2)
        self.assertEqual(summary['word']['with_reference_text'], 1)
        self.assertAlmostEqual(summary['word']['median_asr_disagreement'], .1)
        self.assertAlmostEqual(summary['character']['median_asr_disagreement'], .6)

    def test_selected_quality_uses_fallback_bytes_and_rejects_primary_metrics(self):
        primary = {'id': 'a', 'source_sha256': 'source', 'output_sha256': 'primary',
                   'content': {}, 'secondary_content': {}, 'voice': {}, 'source': '/private/input.wav'}
        case = {'id': 'a', 'source_sha256': 'source', 'output_sha256': 'fallback',
                'fallback_needed': True, 'status': 'verified_negative'}
        measured = dict(primary, output_sha256='fallback', content={'unit': 'word'})
        result = selected_quality_rows([primary], {'a': case}, [measured])
        self.assertEqual(result[0]['output_sha256'], 'fallback')
        self.assertNotIn('source', result[0])
        with self.assertRaisesRegex(ValueError, 'different selected bytes'):
            selected_quality_rows([primary], {'a': case}, [primary])
        with self.assertRaisesRegex(ValueError, 'Incomplete or unexpected'):
            selected_quality_rows([primary], {'a': case}, [])


if __name__ == '__main__':
    unittest.main()
