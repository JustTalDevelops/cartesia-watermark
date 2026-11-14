import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from codec_audio import digest, mix_reconstructions, quantize
from codec_content import assess
from reconstruct_codec import CodecRuntime


class FakeCodec:
    metadata = {'model': 'fixture'}

    def __init__(self):
        self.calls = 0

    def reconstruct(self, audio, rate):
        self.calls += 1
        return audio*.98, 1.


class FakeRecognizer:
    metadata = {'models': 'fixtures'}

    def __init__(self, reject_native=False, supported=True):
        self.reject_native, self.supported = reject_native, supported
        self.source_calls = 0
        self.callback = None

    def transcribe(self, path, language=None):
        if language is None:
            self.source_calls += 1
        if self.callback and language is not None:
            self.callback()
        return {'language': 'en', 'supported': self.supported, 'turbo': ['first', 'second'],
                'mms': ['first', 'changed'] if self.reject_native and path.name == 'native.flac' else ['first', 'second']}


class FakeBand:
    def __init__(self):
        self.closed = False
        self.calls = 0

    def close(self):
        self.closed = True

    def run(self, source, output, preset, metadata, timeout):
        self.calls += 1
        request = json.loads(source.read_text())
        original, rate = sf.read(request['source'], dtype='float32')
        flac = metadata.with_suffix('.flac')
        samples = quantize(flac, original*.97, rate)
        sf.write(output, samples, rate, subtype='FLOAT')
        return {'output_sha256': digest(output), 'source_sha256': request['source_sha256'],
                'native_sha256': request['native_sha256'], 'encoded_flac_sha256': digest(flac)}


class CodecRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root/'source.wav'
        self.output = self.root/'output.wav'
        t = np.arange(5*8000)/8000
        sf.write(self.source, .1*np.sin(2*np.pi*231*t), 8000, subtype='FLOAT')
        self.codec, self.recognizer, self.band = FakeCodec(), FakeRecognizer(), FakeBand()

        def factory(python, script, isolated):
            self.assertEqual(Path(script).name, 'reconstruct_codec_band.py')
            self.assertFalse(isolated)
            return self.band

        self.runtime = CodecRuntime(self.codec, self.recognizer, factory)

    def tearDown(self):
        self.runtime.close()
        self.temporary.cleanup()

    def test_native_acceptance_binds_pcm24_and_never_claims_human_quality(self):
        report = self.runtime.process_file(self.source, self.output)
        samples, rate = sf.read(self.output, dtype='float32')
        self.assertEqual(rate, 8000)
        self.assertTrue(report['content_guard_passed'])
        self.assertEqual(report['paired_disagreement'], {'turbo': 0., 'mms': 0.})
        self.assertFalse(report['quality_verified'])
        self.assertEqual(report['output_sha256'], digest(self.output))
        np.testing.assert_array_equal(samples*2**23, np.rint(samples*2**23))
        self.assertIsNone(self.runtime.cache)
        self.assertFalse(list(self.root.glob('.codec-*')))

    def test_independent_rejection_has_no_output_and_immediate_fallback_reuses_source(self):
        self.recognizer.reject_native = True
        rejected = self.runtime.process_file(self.source, self.output, retain_job=True)
        self.assertFalse(rejected['content_guard_passed'])
        self.assertEqual(rejected['paired_disagreement']['turbo'], 0.)
        self.assertEqual(rejected['paired_disagreement']['mms'], .5)
        self.assertIsNone(rejected['output_sha256'])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.band.calls, 0)
        accepted = self.runtime.process_file(self.source, self.output, 'codec-band')
        self.assertTrue(accepted['content_guard_passed'])
        self.assertTrue(accepted['source_job_reused'])
        self.assertEqual(self.recognizer.source_calls, 1)
        self.assertEqual(self.codec.calls, 1)
        self.assertEqual(self.band.calls, 1)
        self.assertIsNone(self.runtime.cache)

    def test_cache_tampering_fails_before_fallback_or_publication(self):
        self.recognizer.reject_native = True
        self.runtime.process_file(self.source, self.output, retain_job=True)
        self.runtime.cache['native'].write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Cached native'):
            self.runtime.process_file(self.source, self.output, 'codec-band')
        self.assertFalse(self.output.exists())
        self.assertIsNone(self.runtime.cache)
        self.assertEqual(self.band.calls, 0)

    def test_unsupported_language_does_not_run_synthesis_or_write_audio(self):
        self.recognizer.supported = False
        report = self.runtime.process_file(self.source, self.output)
        self.assertFalse(report['content_guard_passed'])
        self.assertEqual(report['rejection_reason'], 'unsupported_language_for_content_guard')
        self.assertEqual(self.codec.calls, 0)
        self.assertFalse(self.output.exists())

    def test_changed_source_stops_before_publication(self):
        self.recognizer.callback = lambda: sf.write(self.source, np.zeros(5*8000), 8000, subtype='FLOAT')
        with self.assertRaisesRegex(RuntimeError, 'changed during content assessment'):
            self.runtime.process_file(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_publication_does_not_remove_a_competing_output(self):
        self.recognizer.callback = lambda: self.output.write_bytes(b'another owner')
        with self.assertRaises(FileExistsError):
            self.runtime.process_file(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b'another owner')

    def test_metadata_race_removes_only_this_jobs_output(self):
        metadata = self.root/'report.json'
        self.recognizer.callback = lambda: metadata.write_text('another owner')
        with self.assertRaises(FileExistsError):
            self.runtime.process_file(self.source, self.output, metadata=metadata)
        self.assertFalse(self.output.exists())
        self.assertEqual(metadata.read_text(), 'another owner')

    def test_close_clears_ephemeral_source_and_stops_owned_worker(self):
        self.runtime.process_file(self.source, self.output, 'codec-band', retain_job=True)
        path = self.runtime.cache['native']
        self.assertTrue(path.exists())
        self.runtime.close()
        self.assertFalse(path.exists())
        self.assertTrue(self.band.closed)


class CodecGuardAndBandTests(unittest.TestCase):
    def test_both_recognizers_must_pass_and_missing_speech_rejects(self):
        before = {'turbo': list(range(100)), 'mms': list(range(100))}
        after = {'turbo': before['turbo'][:-3]+['a', 'b', 'c'], 'mms': before['mms'][:]}
        self.assertTrue(assess(before, after)['content_guard_passed'])
        after['mms'][-4:] = ['a', 'b', 'c', 'd']
        self.assertFalse(assess(before, after)['content_guard_passed'])
        self.assertFalse(assess({'turbo': [], 'mms': []}, {'turbo': [], 'mms': []})['content_guard_passed'])

    def test_runtime_band_matches_independent_research_reference_on_stereo(self):
        from experiments.codec_band_mix import mix_reconstructions as reference
        random = np.random.default_rng(42)
        base = random.normal(0, .05, (5*16000, 2)).astype(np.float32)
        donor = random.normal(0, .05, base.shape).astype(np.float32)
        actual, gain = mix_reconstructions(base, donor, 16000)
        expected, expected_gain = reference(base, donor, 16000, 1000, 4000)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(gain, expected_gain)

    def test_8khz_upper_edge_uses_every_available_frequency_above_lower_edge(self):
        time = np.arange(5*8000)/8000
        low = np.sin(2*np.pi*300*time)
        high = np.sin(2*np.pi*2500*time)
        result, gain = mix_reconstructions((.1*low).astype(np.float32), (.1*high).astype(np.float32), 8000)
        segment = slice(8000, -8000)
        self.assertEqual(gain, 1.)
        self.assertAlmostEqual(2*np.mean(result[segment]*low[segment]), .1, places=3)
        self.assertAlmostEqual(2*np.mean(result[segment]*high[segment]), .1, places=3)


if __name__ == '__main__':
    unittest.main()
