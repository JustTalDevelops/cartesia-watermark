"""Exercise publication boundaries without loading models or making requests."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

import verify_codec_profiles as cli
from speech_repair import pcm_digest


class ProfilePublicationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source, self.output, self.audit = (self.root/name for name in ('in.wav', 'out.flac', 'audit.json'))
        t = np.arange(5*8000)/8000
        sf.write(self.source, .1*np.sin(2*np.pi*213*t), 8000, subtype='FLOAT')

    def detector(self, answers, after_check=None):
        answers = iter(answers)
        self.uploaded = []
        def detect(path):
            self.assertFalse(self.output.exists(), 'Output was exposed before the closing controls')
            self.uploaded.append(path.read_bytes())
            answer = next(answers)
            if after_check:
                after_check(len(self.uploaded))
            return answer if isinstance(answer, tuple) else (200, {'detected': answer})
        return detect

    def fake_models(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        recognition = SimpleNamespace(metadata={}, normalizer=lambda text:text,
            transcribe=Mock(return_value={'supported':True, 'language':'en'}))
        self.runtime = Mock()
        def process(source, output, preset, metadata, retain_job):
            audio, rate = sf.read(source, dtype='float32')
            edited = np.rint(audio*.98*2**23)/2**23
            sf.write(output, edited, rate, subtype='PCM_24')
            evaluated, _ = sf.read(output, dtype='float32')
            return {'preset':preset, 'output_sha256':cli.digest(output), 'content_guard_passed':True,
                    'sample_rate':rate, 'content_guard_sample_grid':'PCM_24',
                    'content_guard_pcm_sha256':pcm_digest(evaluated)}
        self.runtime.process_file.side_effect = process
        def voice(arguments, **kwargs):
            plan = json.loads(Path(arguments[arguments.index('--plan')+1]).read_text())
            self.assertEqual(plan['items'][0]['text'], '')
            Path(arguments[arguments.index('--output')+1]).write_text(json.dumps({'complete':True, 'metric':'fixture'}))
        stack.enter_context(patch.object(cli, 'ContentRecognizer', return_value=recognition))
        stack.enter_context(patch.object(cli, 'CodecRuntime', return_value=self.runtime))
        stack.enter_context(patch.object(cli.subprocess, 'run', side_effect=voice))
        stack.enter_context(patch.object(cli, 'runtime_voice_decision', return_value={'passed':True}))

    def test_negative_original_does_not_load_models_or_publish(self):
        with patch.object(cli, 'ContentRecognizer', side_effect=AssertionError('Unnecessary model loading')):
            result = cli.verify(self.source, self.output, self.audit, self.detector([False, False]))
        self.assertEqual(result['status'], 'original_not_detected')
        self.assertEqual(len(result['checks']), 2)
        self.assertFalse(self.output.exists())

    def test_exact_candidate_published_only_after_four_source_controls(self):
        self.fake_models()
        result = cli.verify(self.source, self.output, self.audit, self.detector([True, True, False, False, True, True]))
        self.assertEqual(self.output.read_bytes(), self.uploaded[2])
        self.assertEqual(self.uploaded[2], self.uploaded[3])
        self.assertEqual(result['output_sha256'], cli.digest(self.output))
        self.assertEqual(result['status'], 'verified_negative')
        self.assertFalse(result['quality_verified'])
        self.runtime.close.assert_called_once()

    def test_conflicting_original_or_candidate_and_failed_closing_controls_never_publish(self):
        self.fake_models()
        for index, answers in enumerate(([True, False], [True, True, False, True],
                                         [True, True, True, False], [True, True, False, False, False, False],
                                         [True, True, (503, {})])):
            with self.subTest(answers=answers), self.assertRaises(RuntimeError):
                cli.verify(self.source, self.output, self.root/f'failure{index}.json', self.detector(answers))
            self.assertFalse(self.output.exists())

    def test_source_mutation_after_opening_checks_never_publishes(self):
        def change(number):
            if number==2:
                audio, rate = sf.read(self.source)
                sf.write(self.source, audio*.99, rate, subtype='FLOAT')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            cli.verify(self.source, self.output, self.audit, self.detector([True, True], change))
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved_before_upload(self):
        self.output.write_bytes(b'existing user file')
        with self.assertRaises(ValueError):
            cli.verify(self.source, self.output, self.audit, lambda _:self.fail('Unexpected upload'))
        self.assertEqual(self.output.read_bytes(), b'existing user file')

    def test_exhausted_profiles_return_no_output_and_keep_rejections(self):
        self.fake_models()
        result = cli.verify(self.source, self.output, self.audit, self.detector([True]*4))
        self.assertEqual(result['status'], 'not_cleared')
        self.assertEqual(result['processing']['attempts'][0]['state'], 'checked')
        self.assertEqual(result['processing']['attempts'][1]['state'], 'duplicate_candidate_skipped')
        self.assertFalse(self.output.exists())
        self.runtime.close.assert_called_once()

    def test_concurrently_created_output_survives_publication_failure(self):
        self.fake_models()
        def competing_writer(number):
            if number==6:
                self.output.write_bytes(b'concurrent user file')
        with self.assertRaises(FileExistsError):
            cli.verify(self.source, self.output, self.audit,
                       self.detector([True, True, False, False, True, True], competing_writer))
        self.assertEqual(self.output.read_bytes(), b'concurrent user file')


if __name__=='__main__':
    unittest.main()
