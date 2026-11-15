"""Exercise latent inventory validation using tiny audio and cached mock evidence."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from experiments import ticodec_latent_generation_report as reporting


class LatentReportingTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        F = root/'codec_profiles_fresh'
        panel = root/'ticodec_latent_development'
        original = root/'ticodec_development'
        for path in (F/'recognition/originals', panel/'recognition/processed', original):
            path.mkdir(parents=True)
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        def audio(path, frequency):
            path.parent.mkdir(parents=True, exist_ok=True)
            samples = np.arange(40000)/8000
            sf.write(path, .1*np.sin(2*np.pi*frequency*samples), 8000, subtype='PCM_24')
            return reporting.digest(path)
        source = root/'source.flac'
        source_sha = audio(source, 230)
        control = panel/'case0/quantized-control.flac'
        control_sha = audio(control, 250)
        cached = {'primary': 'turbo', 'secondary': 'sensevoice', 'language': 'en',
                  'turbo': ['one', 'two'], 'sensevoice': ['one', 'two']}
        before_path = F/'recognition/originals'/(source_sha+'.json')
        save(before_path, cached)
        grid = {'samples': 40000, 'sample_rate': 8000, 'channels': 1}
        old = {'id': 'case0', 'source_sha256': source_sha, 'source_format': grid, 'language': 'en',
               'model': 'fixture', 'voice': 'fixture-voice', 'marked_provenance': True,
               'qualified_marked_transition': False,
               'original_guard_calibration': {'recognition_cache_sha256': reporting.digest(before_path)}}
        save(F/'curated_complete_report.json', {'cases': [old]})
        save(F/'original_guards.json', {})
        save(F/'original_voice.json', {'metric': 'voice-fixture', 'clips': [
            {'id': 'case0', 'original_dnsmos': {'overall': 4.0}}]})
        save(original/'generation.json', {'recognition_configuration_sha256': 'configuration',
             'cases': [{'id': 'case0', 'output_sha256': control_sha}]})
        treatments = []
        for index, (mode, local, global_) in enumerate(reporting.MODES[1:]):
            path = panel/'case0'/(mode+'.flac')
            checksum = audio(path, 270+index*20)
            after = copy.deepcopy(cached)
            if index == 1: after['sensevoice'] = ['changed', 'words']
            cache = panel/'recognition/processed'/(checksum+'.json')
            save(cache, after)
            item = {'mode': mode, 'local_quantized': local, 'global_quantized': global_,
                    'output_file': str(path), 'output_sha256': checksum, 'gain': 1.0,
                    'speech_assessment': reporting.assess(cached, after),
                    'recognition_cache_sha256': reporting.digest(cache),
                    'status': ['processing_eligible', 'content_rejected', 'runtime_quality_rejected'][index]}
            if index != 1:
                speaker = .98 if index == 0 else .90
                receipt_path = panel/'runtime_voice/case0'/(checksum+'.json')
                save(receipt_path, {'complete': True, 'metric': 'voice-fixture', 'clips': [{
                    'status': 'scored', 'original_dnsmos': {'overall': 4.0}, 'source_sha256': source_sha,
                    'output_sha256': checksum, 'speaker_minimum': speaker, 'dnsmos_delta': {'overall': -.1}}]})
                item['runtime_quality_guard'] = {'receipt_sha256': reporting.digest(receipt_path),
                    'source_sha256': source_sha, 'candidate_sha256': checksum, 'speaker_minimum': speaker,
                    'dnsmos_overall_delta': -.1, 'speaker_minimum_required': .95,
                    'dnsmos_overall_delta_required': -.2, 'passed': index == 0}
            treatments.append(item)
        manifest = panel/'generation.json'
        value = {'complete': True, 'completed_study_sha256': reporting.digest(F/'curated_complete_report.json'),
            'original_ticodec_generation_sha256': reporting.digest(original/'generation.json'),
            'planned_spent_sources': 1, 'planned_treatments': 3, 'planned_exact_file_controls': 1,
            'synthetic_control': {'exact_original_adapter_waveform': True, 'maximum_absolute_error': 0},
            'code_files_sha256': {}, 'recognition_configuration_sha256': 'configuration', 'codec': {},
            'cases': [{'id': 'case0', 'source': str(source), 'source_sha256': source_sha, 'language': 'en',
                       'control_file': str(control), 'control_status': 'exact_match',
                       'control_expected_sha256': control_sha, 'control_actual_sha256': control_sha,
                       'treatments': treatments}]}
        save(manifest, value)
        return root, manifest, value

    @contextmanager
    def evidence(self):
        def read(folder, checksum, identity):
            self.assertEqual(identity, 'configuration')
            return json.loads((folder/(checksum+'.json')).read_text())
        with patch.object(reporting, 'original_export'), patch.object(reporting, 'read_recognition', side_effect=read):
            yield

    def test_complete_export_keeps_all_rejections_and_controls_without_claiming_efficacy(self):
        with TemporaryDirectory() as folder, self.evidence():
            root, _, _ = self.fixture(folder)
            public, private = reporting.validated_inventory(root)
            self.assertEqual(public['operational_status_counts'], {
                'processing_eligible': 1, 'content_rejected': 1, 'runtime_quality_rejected': 1})
            self.assertEqual(public['exact_quantized_controls'], 1)
            self.assertEqual([t['detector_eligible'] for t in private['treatments']], [True, False, False])
            self.assertFalse(public['detector_evaluated_by_generation'])
            self.assertNotIn(str(root), json.dumps(public))

    def test_partial_or_missing_treatment_cannot_export(self):
        for mutation in ('partial', 'missing'):
            with self.subTest(mutation=mutation), TemporaryDirectory() as folder, self.evidence():
                root, manifest, value = self.fixture(folder)
                if mutation == 'partial': value['complete'] = False
                else: value['cases'][0]['treatments'].pop()
                manifest.write_text(json.dumps(value))
                with self.assertRaises(ValueError): reporting.validated_inventory(root)

    def test_changed_control_audio_is_rejected(self):
        with TemporaryDirectory() as folder, self.evidence():
            root, _, value = self.fixture(folder)
            control = Path(value['cases'][0]['control_file'])
            audio, rate = sf.read(control)
            sf.write(control, audio*.5, rate, subtype='PCM_24')
            with self.assertRaisesRegex(ValueError, 'control or assessment changed'):
                reporting.validated_inventory(root)

    def test_relaxed_voice_limit_cannot_convert_rejection_to_eligibility(self):
        with TemporaryDirectory() as folder, self.evidence():
            root, manifest, value = self.fixture(folder)
            item = value['cases'][0]['treatments'][2]
            item['runtime_quality_guard'].update(speaker_minimum_required=.85, passed=True)
            item['status'] = 'processing_eligible'
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'quality limits changed'):
                reporting.validated_inventory(root)


if __name__ == '__main__': unittest.main()
