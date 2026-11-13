"""Exercise worker reuse and failure recovery without reconstruction assets."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reconstruction_worker import ReconstructionWorker


FAKE_WORKER = '''\
import json, os, sys, time
print(json.dumps({'ready': True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['preset'] == 'timeout':
        time.sleep(30)
    if request['preset'] == 'crash':
        sys.exit(1)
    if request['preset'] == 'error':
        print(json.dumps({'error': 'invalid audio'}), flush=True)
        continue
    print(json.dumps({'report': {'pid': os.getpid(), 'request': request}}), flush=True)
'''


class ReconstructionWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        script = self.root / 'fake_worker.py'
        script.write_text(FAKE_WORKER)
        self.worker = ReconstructionWorker(sys.executable, script)

    def tearDown(self):
        self.worker.close()
        self.directory.cleanup()

    def run_job(self, preset='standard', timeout=5):
        return self.worker.run(self.root/'input with spaces.wav', self.root/'out.wav',
                               preset, self.root/'metadata.json', timeout)

    def test_reuses_process_and_sends_each_preset(self):
        first = self.run_job()
        second = self.run_job('strong')
        self.assertEqual(first['pid'], second['pid'])
        self.assertEqual(second['request']['preset'], 'strong')
        self.assertEqual(second['request']['input'], str(self.root/'input with spaces.wav'))
        self.assertTrue(Path(second['request']['output']).is_absolute())
        process = self.worker.process
        self.worker.close()
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.worker.close()

    def test_timeout_discards_worker_and_next_job_recovers(self):
        self.run_job()
        process = self.worker.process
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_job('timeout', timeout=.05)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.worker.process)
        self.assertIn('pid', self.run_job())

    def test_crash_and_reported_error_discard_worker(self):
        for preset in ('crash', 'error'):
            with self.assertRaises(RuntimeError):
                self.run_job(preset)
            self.assertIsNone(self.worker.process)
            self.assertIn('pid', self.run_job())

    def test_startup_failure_is_detected(self):
        self.worker.command = [sys.executable, '-c', 'raise SystemExit(1)']
        with self.assertRaises(RuntimeError):
            self.run_job()
        self.assertIsNone(self.worker.process)

    def test_startup_timeout_is_detected(self):
        self.worker.command = [sys.executable, '-c', 'import time; time.sleep(30)']
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_job(timeout=.05)
        self.assertIsNone(self.worker.process)


class ReconstructionFileTests(unittest.TestCase):
    def test_file_safety_and_metadata(self):
        import numpy as np
        import soundfile as sf
        import torch
        from reconstruct_audio import process_file

        class Model:
            def configurate_(self, **settings):
                self.settings = settings

        def inference(model, audio, rate, device, **kwargs):
            # A deterministic stand-in verifies channel, rate, and file handling.
            return torch.as_tensor(audio), rate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, metadata = root/'input.wav', root/'output.wav', root/'meta.json'
            audio = np.full((321, 2), .1, dtype=np.float32)
            sf.write(source, audio, 48000, subtype='FLOAT')
            model = Model()
            for target, meta in ((source, None), (output, source), (root/'out.mp3', None)):
                with self.assertRaises(ValueError):
                    process_file(model, inference, source, target, 'standard', meta)
            report = process_file(model, inference, source, output, 'strong', metadata)
            actual, rate = sf.read(output)
            np.testing.assert_array_equal(actual, audio)
            self.assertEqual(rate, 48000)
            self.assertEqual(report['channels'], 2)
            self.assertEqual(model.settings['lambd'], .9)
            self.assertEqual(json.loads(metadata.read_text()), report)
            with self.assertRaises(ValueError):
                process_file(model, inference, source, output, 'standard')


if __name__ == '__main__':
    unittest.main()
