"""Verify exact-file publication and failure handling without external calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import verify_reconstruction
from verify_reconstruction import NotCleared, digest, reconstruct_verified


class FakeWorker:
    def __init__(self):
        self.calls = []
        self.closed = False

    def close(self):
        self.closed = True

    def run(self, source, output, preset, metadata_path, timeout):
        audio, sample_rate = sf.read(source)
        # Distinct candidate bytes make accidental publication of a rejected
        # preset observable, including when FLAC is the requested final format.
        edited = audio * (.98 if preset == 'full-flow' else .96)
        sf.write(output, edited, sample_rate, subtype='FLOAT')
        metadata = {'preset': preset, 'output_sha256': digest(output), 'seed': 2035}
        self.calls.append(preset)
        return metadata


class VerifiedReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source, self.output, self.report = (self.root/'source.wav', self.root/'out.wav', self.root/'audit.json')
        t = np.arange(5*8000)/8000
        sf.write(self.source, .1*np.sin(2*np.pi*213*t), 8000, subtype='FLOAT')
        self.worker = FakeWorker()

    def tearDown(self):
        self.directory.cleanup()

    def run_verified(self, detector, **kwargs):
        return reconstruct_verified(self.source, self.output, self.report,
                                    worker=self.worker, detector=detector, **kwargs)

    def test_final_file_is_the_exact_checked_blob(self):
        checked = []

        def detector(candidate):
            checked.append(candidate.read_bytes())
            return 200, {'detected': False}

        source_hash = digest(self.source)
        result = self.run_verified(detector)
        self.assertEqual(self.output.read_bytes(), checked[0])
        self.assertEqual(result['output_sha256'], digest(self.output))
        self.assertEqual(digest(self.source), source_hash)
        self.assertEqual(result['status'], 'verified_negative')
        self.assertFalse(result['quality_verified'])

    def test_owned_runtimes_are_loaded_lazily_reused_and_closed(self):
        for rejected in (0, 2):
            with self.subTest(rejected=rejected):
                workers, scripts = [], []

                def factory(python, script, **kwargs):
                    scripts.append(Path(script).name)
                    workers.append(FakeWorker())
                    return workers[-1]

                answers = iter([(200, {'detected': True})]*rejected+[(200, {'detected': False})])
                with patch('verify_reconstruction.ReconstructionWorker', side_effect=factory):
                    result = reconstruct_verified(
                        self.source, self.root/f'checked{rejected}.wav', self.root/f'checked{rejected}.json',
                        presets=('smoothed-vocoder', 'full-flow', 'strong'), detector=lambda _: next(answers))
                self.assertEqual(scripts, ['reconstruct_vocoder.py']+
                                 (['reconstruct_audio.py'] if rejected else []))
                self.assertEqual(workers[0].calls, ['smoothed-vocoder'])
                if rejected:
                    self.assertEqual(workers[1].calls, ['full-flow', 'strong'])
                self.assertTrue(all(w.closed for w in workers))
                self.assertEqual(result['selected_preset'], 'strong' if rejected else 'smoothed-vocoder')

    def test_missing_api_key_fails_before_reconstruction(self):
        with patch.dict('os.environ', {}), patch('cartesia._env', return_value=None):
            with self.assertRaisesRegex(ValueError, 'CARTESIA_API_KEY'):
                reconstruct_verified(self.source, self.output, self.report, worker=self.worker)
        self.assertEqual(self.worker.calls, [])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_interrupted_copy_leaves_no_partial_output(self):
        def interrupted(source, target):
            target.write(source.read(20))
            raise KeyboardInterrupt

        with patch('verify_reconstruction.shutil.copyfileobj', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_verified(lambda _: (200, {'detected': False}))
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.out.wav.*.tmp')), [])

    def test_output_created_during_the_api_check_is_preserved(self):
        def detector(_):
            self.output.write_bytes(b'Other process output')
            return 200, {'detected': False}

        with self.assertRaises(FileExistsError):
            self.run_verified(detector)
        self.assertEqual(self.output.read_bytes(), b'Other process output')

    def test_rejected_preset_is_not_published(self):
        checked = []

        def detector(candidate):
            checked.append(candidate.read_bytes())
            return 200, {'detected': len(checked) == 1}

        result = self.run_verified(detector)
        self.assertEqual(self.worker.calls, ['full-flow', 'strong'])
        self.assertEqual(result['selected_preset'], 'strong')
        self.assertNotEqual(checked[0], checked[1])
        self.assertEqual(self.output.read_bytes(), checked[1])

    def test_no_output_when_all_candidates_are_positive(self):
        with self.assertRaises(NotCleared):
            self.run_verified(lambda _: (200, {'detected': True}))
        self.assertFalse(self.output.exists())
        report = json.loads(self.report.read_text())
        self.assertEqual(report['status'], 'not_cleared')
        self.assertEqual(len(report['attempts']), 2)

    def test_confirmation_rechecks_the_same_bytes_before_publication(self):
        checked = []

        def detector(candidate):
            self.assertFalse(self.output.exists())
            checked.append(candidate.read_bytes())
            return 200, {'detected': False}

        result = self.run_verified(detector, confirmations=2)
        self.assertEqual(checked, [self.output.read_bytes()]*2)
        self.assertEqual(result['negative_checks'], 2)
        self.assertEqual(result['required_negative_checks'], 2)
        self.assertEqual(len(result['attempts'][0]['checks']), 2)

    def test_later_positive_rejects_an_earlier_negative_and_uses_fallback(self):
        answers = iter((False, True, False, False))
        result = self.run_verified(lambda _: (200, {'detected': next(answers)}), confirmations=2)
        self.assertEqual(result['selected_preset'], 'strong')
        self.assertEqual([a['detected'] for a in result['attempts']], [True, False])
        self.assertEqual(result['negative_checks'], 2)

    def test_unavailable_confirmation_publishes_nothing(self):
        answers = iter(((200, {'detected': False}), (503, {})))
        with self.assertRaises(RuntimeError):
            self.run_verified(lambda _: next(answers), confirmations=2)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.worker.calls, ['full-flow'])

    def test_confirmation_count_is_bounded_before_processing(self):
        for count in (0, 4, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.run_verified(lambda _: (200, {'detected': False}), confirmations=count)
        self.assertEqual(self.worker.calls, [])
        self.assertFalse(self.report.exists())

    def test_api_errors_and_malformed_replies_do_not_trigger_more_presets(self):
        for status, response in ((429, {}), (503, {}), (200, {'detected': 0}), (200, {})):
            with self.subTest(status=status, response=response):
                self.report.unlink(missing_ok=True)
                self.worker.calls.clear()
                with self.assertRaises(RuntimeError):
                    self.run_verified(lambda _, s=status, r=response: (s, r))
                self.assertFalse(self.output.exists())
                self.assertEqual(self.worker.calls, ['full-flow'])
                self.assertEqual(json.loads(self.report.read_text())['status'], 'verification_failed')

    def test_changed_candidate_is_rejected_even_on_a_negative_response(self):
        def detector(candidate):
            candidate.write_bytes(candidate.read_bytes()+b'changed')
            return 200, {'detected': False}

        with self.assertRaisesRegex(RuntimeError, 'Candidate changed'):
            self.run_verified(detector)
        self.assertFalse(self.output.exists())

    def test_unavailable_verification_never_publishes_or_tries_more_presets(self):
        def unavailable(_):
            raise ConnectionError('Simulated unavailable detector')

        with self.assertRaises(ConnectionError):
            self.run_verified(unavailable)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.worker.calls, ['full-flow'])
        self.assertEqual(json.loads(self.report.read_text())['status'], 'verification_failed')

    def test_changed_source_is_rejected(self):
        def detector(_):
            self.source.write_bytes(self.source.read_bytes()+b'changed')
            return 200, {'detected': False}

        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            self.run_verified(detector)
        self.assertFalse(self.output.exists())

    def test_flac_is_encoded_before_verification_and_copied_without_reencoding(self):
        self.output = self.output.with_suffix('.flac')
        checked = []

        def detector(candidate):
            self.assertEqual(candidate.suffix, '.flac')
            self.assertEqual(sf.info(candidate).subtype, 'PCM_24')
            checked.append(candidate.read_bytes())
            return 200, {'detected': False}

        self.run_verified(detector)
        self.assertEqual(self.output.read_bytes(), checked[0])
        self.assertEqual(sf.info(self.output).frames, sf.info(self.source).frames)

    def test_oversize_output_is_never_sent(self):
        with patch('verify_reconstruction.MAX_BYTES', 1):
            with self.assertRaisesRegex(ValueError, '32 MB'):
                self.run_verified(lambda _: self.fail('Oversize file was sent'))
        self.assertFalse(self.output.exists())

    def test_existing_audio_and_reports_are_protected(self):
        self.output.write_bytes(b'existing output')
        with self.assertRaises(ValueError):
            self.run_verified(lambda _: self.fail('Request was sent'))
        self.assertEqual(self.output.read_bytes(), b'existing output')
        self.output.unlink()
        self.report.write_text('existing report')
        with self.assertRaises(ValueError):
            self.run_verified(lambda _: self.fail('Request was sent'))
        self.assertEqual(self.report.read_text(), 'existing report')
        self.assertEqual(self.worker.calls, [])

    def test_final_audit_failure_removes_the_new_output(self):
        save = verify_reconstruction.save_report

        def failing_final_report(path, report, create=False):
            if report['status'] == 'verified_negative':
                raise OSError('Simulated final audit failure')
            return save(path, report, create)

        with patch('verify_reconstruction.save_report', side_effect=failing_final_report):
            with self.assertRaises(OSError):
                self.run_verified(lambda _: (200, {'detected': False}))
        self.assertFalse(self.output.exists())
        self.assertEqual(json.loads(self.report.read_text())['status'], 'verification_failed')

    def test_format_damage_is_rejected_before_any_request(self):
        def damaged(source, output, preset, metadata, timeout):
            sf.write(output, np.zeros(100), 8000, subtype='FLOAT')
            return {'preset': preset, 'output_sha256': digest(output)}

        with patch.object(self.worker, 'run', side_effect=damaged):
            with self.assertRaisesRegex(RuntimeError, 'format'):
                self.run_verified(lambda _: self.fail('Damaged recording was sent'))
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
