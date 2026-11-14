import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from experiments.generate_codec_study import Ledger, canonicalize


class CodecStudyGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root/'budget.json'
        self.old = {'id': 'earlier', 'reserved_credits': 100, 'status': 'completed'}
        self.budget = {'authorized_cap_usd': 50, 'reservation_rate_usd_per_million_credits': 65,
                       'maximum_reserved_credits': 769230, 'reserved_credits': 100, 'requests': [self.old]}
        self.path.write_text(json.dumps(self.budget))
        self.ledger = Ledger(self.root)
        self.item = {'id': 'new', 'model': 'sonic-3.6', 'voice': 'public-fixture', 'text': 'é'}

    def tearDown(self):
        self.temporary.cleanup()

    def test_reservation_preserves_prior_usage_and_counts_utf8_conservatively(self):
        self.ledger.reserve(self.item)
        after = json.loads(self.path.read_text())
        self.assertEqual(after['requests'][0], self.old)
        self.assertEqual(after['requests'][1]['reserved_credits'], 4)
        self.assertEqual(after['reserved_credits'], 104)
        self.assertFalse(after['requests'][1]['attempted'])

    def test_attempted_or_existing_requests_are_never_reserved_again(self):
        self.ledger.reserve(self.item)
        self.ledger.update('new', attempted=True, status='failed_or_uncertain')
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'prior reservation'):
            self.ledger.reserve(self.item)
        self.assertEqual(before, self.path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.ledger.update('new', reserved_credits=0)

    def test_budget_exhaustion_stops_without_rewriting_reservations(self):
        self.budget['maximum_reserved_credits'] = 101
        self.path.write_text(json.dumps(self.budget))
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'limit would be exceeded'):
            self.ledger.reserve(self.item)
        self.assertEqual(before, self.path.read_bytes())

    def test_canonical_source_keeps_the_whole_grid_and_separates_stereo_gains(self):
        source, output = self.root/'raw.wav', self.root/'canonical.flac'
        audio = np.full(5*8000, .1, dtype=np.float32)
        sf.write(source, audio, 8000, subtype='FLOAT')
        info = canonicalize(source, output, {'sample_rate': 8000, 'stereo_derivation_gains': [.9, .63]})
        result, rate = sf.read(output, dtype='float32')
        self.assertEqual((rate, result.shape), (8000, (len(audio), 2)))
        self.assertEqual(info['samples'], len(audio))
        np.testing.assert_allclose(result[0], [.09, .063], atol=2**-23)

    def test_short_long_slot_is_recorded_as_failure_instead_of_padding(self):
        source, output = self.root/'raw.wav', self.root/'canonical.flac'
        sf.write(source, np.zeros(5*8000), 8000, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'long-duration slot'):
            canonicalize(source, output, {'sample_rate': 8000, 'expected_duration_seconds': [100, 220]})
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
