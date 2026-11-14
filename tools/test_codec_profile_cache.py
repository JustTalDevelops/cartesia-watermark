from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.codec_profile_cache import StudyRecognizer, read_recognition, store_recognition
from codec_audio import digest


class ProfileRecognitionCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root/'audio'; self.audio.write_bytes(b'fixture')
        self.real = Mock(metadata={'fixture': True}, normalizer=str.lower)
        self.rec = StudyRecognizer(self.root/'originals', self.root/'candidates', self.real)

    def test_missing_original_calibration_cannot_silently_run_or_use_other_data(self):
        with self.assertRaises(FileNotFoundError):
            self.rec.transcribe(self.audio)
        self.real.transcribe.assert_not_called()

    def test_original_identity_is_bound_to_file_and_configuration(self):
        before = {'language': 'en', 'turbo': ['recognized']}
        store_recognition(self.rec.source_directory, digest(self.audio), self.rec.identity, before)
        self.assertEqual(self.rec.transcribe(self.audio), before)
        with self.assertRaises(ValueError):
            read_recognition(self.rec.source_directory, digest(self.audio), 'different configuration')
        self.audio.write_bytes(b'changed')
        with self.assertRaises(FileNotFoundError):
            self.rec.transcribe(self.audio)

    def test_candidate_uses_real_recognition_and_cannot_overwrite_a_different_observation(self):
        result = {'language': 'en', 'turbo': ['candidate']}
        self.real.transcribe_candidate.return_value = result
        self.assertEqual(self.rec.transcribe_candidate(self.audio, {'language': 'en'}), result)
        self.real.transcribe_candidate.assert_called_once()
        self.assertEqual(read_recognition(self.rec.candidate_directory, digest(self.audio), self.rec.identity), result)
        with self.assertRaises(ValueError):
            store_recognition(self.rec.candidate_directory, digest(self.audio), self.rec.identity, {'changed': True})


if __name__ == '__main__':
    unittest.main()
