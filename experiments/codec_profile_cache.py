"""Private, file-bound recognition caches for the prospective profile study.

Only normalized recognition and model/source identity enter processing caches.
Reference text and quality scores belong to separate evaluation reports.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'tools'))
from codec_audio import digest
from codec_content import ContentRecognizer

IDENTITY_FILES = ('codec_content.py', 'codec_text.py', 'codec_assets.json', 'speech_repair_assets.json',
                  'korean_spacing.py', 'whisper_encoder_reuse.py', 'speech_repair.py')


def configuration(metadata):
    value = {'metadata': metadata, 'source_files_sha256': {n: digest(ROOT/'tools'/n) for n in IDENTITY_FILES}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def store_recognition(directory, checksum, identity, recognition):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(checksum+'.json')
    value = {'audio_sha256': checksum, 'configuration_sha256': identity, 'recognition': recognition}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('A recognition cache would change an existing observation')
        return path
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def read_recognition(directory, checksum, identity):
    value = json.loads((directory/(checksum+'.json')).read_text())
    if value['audio_sha256'] != checksum or value['configuration_sha256'] != identity:
        raise ValueError('Source or recognition configuration differs from its original calibration')
    return value['recognition']


class StudyRecognizer:
    """Reuse required original-only preflight; compute and retain real candidates."""
    def __init__(self, source_directory, candidate_directory, recognizer=None):
        self.recognizer = recognizer if recognizer is not None else ContentRecognizer()
        self.metadata = self.recognizer.metadata
        self.normalizer = self.recognizer.normalizer
        self.identity = configuration(self.metadata)
        self.source_directory, self.candidate_directory = Path(source_directory), Path(candidate_directory)

    def transcribe(self, path, language=None):
        if language is None:
            return read_recognition(self.source_directory, digest(path), self.identity)
        result = self.recognizer.transcribe(path, language)
        store_recognition(self.candidate_directory, digest(path), self.identity, result)
        return result

    def transcribe_candidate(self, path, before):
        checksum = digest(path)
        result = self.recognizer.transcribe_candidate(path, before)
        if digest(path) != checksum:
            raise ValueError('Candidate changed during recognition')
        store_recognition(self.candidate_directory, checksum, self.identity, result)
        return result

    def secondary_transcribe(self, path, language):
        return self.recognizer.secondary_transcribe(path, language)
