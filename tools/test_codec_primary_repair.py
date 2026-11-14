from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.codec_primary_repair import primary_regions


class PrimaryRepairTests(unittest.TestCase):
    def setUp(self):
        self.before = [str(i) for i in range(40)]
        self.after = self.before.copy()
        self.after[20] = 'changed'
        self.timed = [(u, (i*.1, i*.1+.08)) for i, u in enumerate(self.before)]

    def propose(self, **kwargs):
        return primary_regions(self.before, self.after, self.timed, self.timed, self.timed, 5., **kwargs)

    def test_primary_change_selects_bounded_interval_even_when_secondary_is_unchanged(self):
        regions, proof = self.propose()
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0][0], 1.88)
        self.assertAlmostEqual(regions[0][1], 2.2)
        self.assertFalse(proof['reference_text_used'])

    def test_insertion_uses_both_adjacent_timing_anchors(self):
        self.after = self.before[:20]+['inserted']+self.before[20:]
        regions, _ = self.propose()
        self.assertLess(regions[0][0], 1.9)
        self.assertGreater(regions[0][1], 2.08)

    def test_donor_error_in_the_interval_or_padding_is_rejected(self):
        donor = self.timed.copy()
        donor[19] = ('wrong', donor[19][1])
        with self.assertRaisesRegex(ValueError, 'No uniquely'):
            primary_regions(self.before, self.after, self.timed, donor, self.timed, 5.)

    def test_ambiguous_context_and_unanchored_end_changes_are_rejected(self):
        self.before = ['repeated']*40
        self.after = self.before.copy(); self.after[20] = 'changed'
        self.timed = [(u, (i*.1, i*.1+.08)) for i, u in enumerate(self.before)]
        with self.assertRaisesRegex(ValueError, 'No uniquely'):
            self.propose()
        self.after = self.before.copy(); self.after[0] = 'changed'
        with self.assertRaisesRegex(ValueError, 'No uniquely'):
            self.propose()

    def test_missing_timestamps_and_donor_drift_cannot_select_audio(self):
        bad = self.timed.copy(); bad[20] = ('20', (None, None))
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            primary_regions(self.before, self.after, self.timed, bad, self.timed, 5.)
        unordered = self.timed.copy(); unordered[20], unordered[21] = unordered[21], unordered[20]
        with self.assertRaisesRegex(ValueError, 'timing order'):
            primary_regions(self.before, self.after, self.timed, unordered, self.timed, 5.)
        shifted = [(u, (l+.4, r+.4)) for u, (l, r) in self.timed]
        with self.assertRaisesRegex(ValueError, 'No uniquely'):
            primary_regions(self.before, self.after, self.timed, shifted, self.timed, 5.)


if __name__ == '__main__':
    unittest.main()
