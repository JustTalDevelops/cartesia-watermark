"""Ensure follow-up inventories retain unsuccessful and unrendered repairs."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.check_codec_followups import validate_repair_inventory


def fixtures():
    rows = [{'id': str(i), 'source': 'source-'+str(i), 'proposals': [
        {'donor': donor, 'donor_file': donor+'-'+str(i), 'status': 'planned'}
        for donor in ('native-zero', 'continuous-local')]} for i in range(25)]
    declared = {'cases': deepcopy(rows), 'planned_sources': 25, 'planned_slots': 50,
        'maximum_coverage': .20, 'maximum_regions': 8, 'padding_seconds': .12,
        'timing_drift_limit_seconds': .25, 'reference_text_used_for_generation': False,
        'detector_labels_used_for_proposals': False, 'original_waveform_copied': False}
    for row in rows:
        for proposal in row['proposals']: proposal['status'] = 'not_needed_speech_guard_passed'
    generated = {'complete': True, 'cases': deepcopy(rows)}
    first = generated['cases'][0]['proposals'][0]
    first.update(status='rendered_requires_speech_and_voice_checks', output_file='generated.flac')
    measured = {'complete': True, 'cases': deepcopy(generated['cases']),
        'planned_sources': 25, 'planned_slots': 50, 'expected_rendered_outputs': 1,
        'reference_text_used_for_generation_or_selection': False, 'new_http_requests': 0,
        'new_tts_requests': 0, 'fresh_efficacy_sources': 0, 'detector_checked': False,
        'independent_quality_evaluated': False}
    measured['cases'][0]['proposals'][0].update(status='content_rejected',
        proposal_status='rendered_requires_speech_and_voice_checks')
    return measured, generated, declared, [str(i) for i in range(25)]


class InventoryTests(unittest.TestCase):
    def test_failed_and_unrendered_rows_are_preserved(self):
        validate_repair_inventory(*fixtures())

    def test_omitted_source_is_rejected(self):
        values = fixtures(); values[0]['cases'].pop()
        with self.assertRaises(ValueError): validate_repair_inventory(*values)

    def test_missing_donor_is_rejected(self):
        values = fixtures(); values[0]['cases'][0]['proposals'].pop()
        with self.assertRaises(ValueError): validate_repair_inventory(*values)

    def test_changed_source_or_output_is_rejected(self):
        for field in ('source', 'output_file'):
            with self.subTest(field=field):
                values = fixtures(); row = values[0]['cases'][0]
                if field == 'source': row[field] = 'other'
                else: row['proposals'][0][field] = 'other.flac'
                with self.assertRaises(ValueError): validate_repair_inventory(*values)

    def test_unfinished_measurement_is_rejected(self):
        values = fixtures(); values[0]['cases'][0]['proposals'][0]['status'] = 'processing'
        with self.assertRaises(ValueError): validate_repair_inventory(*values)

    def test_unrendered_slot_cannot_become_eligible(self):
        values = fixtures(); values[0]['cases'][1]['proposals'][0]['status'] = 'processing_eligible'
        with self.assertRaises(ValueError): validate_repair_inventory(*values)

    def test_original_audio_or_reference_guidance_is_rejected(self):
        for key in ('original_waveform_copied', 'reference_text_used_for_generation',
                    'detector_labels_used_for_proposals'):
            with self.subTest(key=key):
                values = fixtures(); values[2][key] = True
                with self.assertRaises(ValueError): validate_repair_inventory(*values)

    def test_weaker_coverage_limit_is_rejected(self):
        values = fixtures(); values[2]['maximum_coverage'] = .30
        with self.assertRaises(ValueError): validate_repair_inventory(*values)


if __name__ == '__main__': unittest.main()
