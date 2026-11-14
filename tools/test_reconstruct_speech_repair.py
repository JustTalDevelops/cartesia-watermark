"""Exercise guarded publication without neural assets or detector requests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from reconstruct_speech_repair import SpeechRepairRuntime, digest
from speech_repair import pcm_digest


class Recognizer:
    normalizer = staticmethod(str.lower)
    metadata = {'model': 'fixture recognizer'}

    def __init__(self, mode='repair'):
        self.mode = mode

    def transcribe(self, path, language=None):
        original = path.name == 'source.wav'
        matched = original or self.mode == 'unchanged' or (path.name == 'repaired.flac' and self.mode != 'reject')
        word = 'world' if matched else 'friend'
        interval = (0., 4.9) if self.mode == 'large' else (2., 2.5 if matched else 2.6)
        return {'text': 'hello '+word, 'chunks': [{'text': 'hello', 'timestamp': (1., 1.8)},
                                                {'text': word, 'timestamp': interval}]}, 'en'


class Worker:
    def __init__(self, script):
        self.script = script
        self.closed = False
        self.calls = []

    def run(self, source, output, preset, metadata, timeout):
        self.calls.append(preset)
        manifest = json.loads(source.read_text()) if preset == 'regional-vocoder' else None
        audio, rate = sf.read(Path(manifest['source']) if manifest else source, dtype='float32')
        sf.write(output, audio*(.9 if manifest else .8), rate, subtype='FLOAT')
        return {'preset': preset, 'output_sha256': digest(output), **({
            'source_sha256': manifest['source_sha256'], 'valid_regions_seconds': manifest['regions_seconds'],
            'waveform_is_partial': True} if manifest else {})}

    def close(self):
        self.closed = True


class SpeechRepairRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.source = self.directory/'source.wav'
        self.output = self.directory/'output.wav'
        t = np.arange(40000, dtype=np.float32)/8000
        sf.write(self.source, .1*np.sin(t*213*2*np.pi), 8000, subtype='FLOAT')
        self.workers = []

    def tearDown(self):
        self.temporary.cleanup()

    def runtime(self, mode='repair'):
        def factory(python, script, **kwargs):
            self.assertFalse(kwargs['isolated'])
            worker = Worker(Path(script).name)
            self.workers.append(worker)
            return worker
        return SpeechRepairRuntime(Recognizer(mode), factory)

    def test_repaired_publication_is_quantized_and_contains_no_transcript(self):
        original = self.source.read_bytes()
        runtime = self.runtime()
        try:
            report = runtime.process_file(self.source, self.output)
        finally:
            runtime.close()
        self.assertTrue(report['content_guard_passed'])
        self.assertTrue(report['repair_applied'])
        self.assertEqual(report['asr_final_disagreement'], 0)
        self.assertEqual(self.source.read_bytes(), original)
        audio, rate = sf.read(self.output, dtype='float32')
        self.assertEqual(rate, 8000)
        self.assertEqual(pcm_digest(audio), report['content_guard_pcm_sha256'])
        np.testing.assert_array_equal(audio*2**23, np.rint(audio*2**23))
        for transcript in ('hello world', 'hello friend'):
            self.assertNotIn(transcript, json.dumps(report))
        self.assertEqual(list(self.directory.glob('.speech-repair-*')), [])
        self.assertTrue(all(w.closed for w in self.workers))

    def test_unchanged_recognition_does_not_load_donor(self):
        runtime = self.runtime('unchanged')
        try:
            report = runtime.process_file(self.source, self.output)
        finally:
            runtime.close()
        self.assertTrue(report['content_guard_passed'])
        self.assertFalse(report['repair_applied'])
        self.assertEqual(len(self.workers), 1)

    def test_failed_repair_publishes_metadata_without_audio(self):
        runtime = self.runtime('reject')
        metadata = self.directory/'rejected.json'
        try:
            report = runtime.process_file(self.source, self.output, metadata=metadata)
        finally:
            runtime.close()
        self.assertFalse(report['content_guard_passed'])
        self.assertFalse(self.output.exists())
        self.assertIsNone(report['output_sha256'])
        self.assertEqual(json.loads(metadata.read_text()), report)

    def test_oversized_repair_is_rejected_before_loading_donor(self):
        runtime = self.runtime('large')
        try:
            report = runtime.process_file(self.source, self.output)
        finally:
            runtime.close()
        self.assertFalse(report['content_guard_passed'])
        self.assertEqual(report['rejection_reason'], 'RepairTooLarge')
        self.assertFalse(self.output.exists())
        self.assertEqual(len(self.workers), 1)

    def test_existing_output_is_preserved(self):
        self.output.write_bytes(b'Existing result')
        runtime = self.runtime()
        try:
            with self.assertRaises(ValueError):
                runtime.process_file(self.source, self.output)
        finally:
            runtime.close()
        self.assertEqual(self.output.read_bytes(), b'Existing result')
        self.assertEqual(self.workers, [])


if __name__ == '__main__':
    unittest.main()
