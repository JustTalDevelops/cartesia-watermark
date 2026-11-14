from pathlib import Path
import unittest
from unittest.mock import Mock

from codec_content import ContentRecognizer, assess


class CandidateGuardOrder(unittest.TestCase):
    def setUp(self):
        self.rec = ContentRecognizer.__new__(ContentRecognizer)
        self.order = []
        self.rec.features = lambda path: ([], {})
        self.rec.normalizer = str.lower
        self.original = ['word'+str(i) for i in range(100)]
        self.secondary = self.original.copy()
        self.rec.whisper_model = Mock()
        self.rec.whisper_model.generate.side_effect = lambda **kwargs: self.order.append('primary') or [[1]]
        self.rec.whisper_processor = Mock()
        self.rec.whisper_processor.batch_decode.return_value = [' '.join(self.original)]
        self.rec._large = Mock(side_effect=AssertionError('Large should not be loaded for an early rejection'))
        self.rec.secondary_transcribe = Mock(side_effect=self.transcribe_secondary)
        self.before = {'language': 'en', 'supported': True, 'primary': 'turbo', 'secondary': 'sensevoice',
                       'turbo': self.original, 'sensevoice': self.original}

    def transcribe_secondary(self, path, language, audio):
        self.order.append('secondary')
        return self.before['secondary'], self.secondary, {'duration': 5., 'chunks': []}

    def test_failed_secondary_skips_primary_and_cannot_pass(self):
        self.secondary[-4:] = ['changed']*4
        after = self.rec.transcribe_candidate(Path('fixture'), self.before)
        self.assertEqual(self.order, ['secondary'])
        self.assertTrue(after['primary_skipped'])
        scored = assess(self.before, after)
        self.assertFalse(scored['content_guard_passed'])
        self.assertIsNone(scored['paired_disagreement']['turbo'])
        self.assertEqual(scored['paired_disagreement']['sensevoice'], .04)

    def test_boundary_pass_still_runs_both_once_and_matches_full_transcription(self):
        self.secondary[-3:] = ['changed']*3
        after = self.rec.transcribe_candidate(Path('fixture'), self.before)
        self.assertEqual(self.order, ['secondary', 'primary'])
        self.assertTrue(assess(self.before, after)['content_guard_passed'])
        self.order.clear()
        full = self.rec.transcribe(Path('fixture'), 'en')
        self.assertEqual(self.order, ['primary', 'secondary'])
        self.assertEqual(after, full)

    def test_tamil_uses_its_existing_one_percent_limit_before_loading_large(self):
        self.before.update(language='ta', primary='large', secondary='mms',
                           large=self.original, mms=self.original)
        self.secondary[-2:] = ['changed']*2
        after = self.rec.transcribe_candidate(Path('fixture'), self.before)
        self.assertEqual(self.order, ['secondary'])
        self.rec._large.assert_not_called()
        self.assertFalse(assess(self.before, after)['content_guard_passed'])

    def test_missing_speech_and_mismatched_profiles_cannot_skip_into_acceptance(self):
        self.secondary = []
        after = self.rec.transcribe_candidate(Path('fixture'), self.before)
        self.assertFalse(assess(self.before, after)['content_guard_passed'])
        self.before['secondary'] = 'mms'
        with self.assertRaisesRegex(ValueError, 'same source language profile'):
            self.rec.transcribe_candidate(Path('fixture'), self.before)


if __name__ == '__main__':
    unittest.main()
