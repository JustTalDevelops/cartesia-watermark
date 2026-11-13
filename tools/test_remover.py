"""Small CPU checks for the audio editor and file-safety guarantees.

    .venv/bin/python -m unittest discover -s tools -p 'test_remover.py'
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from remover import Remover, load_model, process_audio, quality_metrics, native_spectral_metrics, resample
import remove_watermark
import verify_remover


class RemoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1)
        self.model = Remover()

    def test_identity_initialization_and_input_gradients(self):
        x = (torch.randn(2, 16000) * 0.1).requires_grad_()
        y = self.model(x)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(self.model.net[-1].weight.grad.abs().max().item(), 0)

    def test_training_temporal_pool_preserves_remaining_positive_sections(self):
        import train_remover

        class SectionHead(torch.nn.Module):
            def forward(self, hidden):
                return hidden[:, 0, 0]

        # A short positive section is obscured by many negative sections in
        # a mean score. The optional training pool must retain that signal.
        hidden = torch.tensor([[[3.], [-4.], [-4.], [-4.], [-4.]]], requires_grad=True)
        speech = torch.ones(1, 5*320)*.1
        with patch.object(train_remover.th, 'WIN', 1), patch.object(train_remover.th, 'HOP', 1):
            mean = train_remover.Teacher.pool(hidden, speech, [SectionHead()])
            worst = train_remover.Teacher.pool(hidden, speech, [SectionHead()], 'smoothmax')
            self.assertLess(float(mean.detach()), 0)
            self.assertGreater(float(worst.detach()), 0)
            worst.backward()
            self.assertTrue(torch.isfinite(hidden.grad).all())
            self.assertGreater(float(hidden.grad[0, 0]), .9)
            silent = train_remover.Teacher.pool(hidden.detach(), torch.zeros_like(speech),
                                                [SectionHead()], 'smoothmax')
            torch.testing.assert_close(silent, torch.zeros_like(silent))

    def test_training_keeps_quality_feasible_initial_checkpoint(self):
        import train_remover

        class DummyTeacher(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.threshold = self.margin = 0.
                self.waveform_weight = self.control_waveform_weight = self.spectral_weight = 1.
                self.cross_encoder = self.intelligibility = self.perceptual = None
                self.detector_config = self.member_hashes = {}

            def forward(self, audio):
                return audio.square().mean(-1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {'train': [{'marked': True, 'voice': 'train'}]*4,
                    'validation': [{'marked': True, 'voice': 'val'}]*4}
            initial = {'objective': 2., 'min_proxy_pesq': 4.6}
            damaged = {'objective': 1., 'min_proxy_pesq': 4.0}
            audio = [np.ones(3*16000, dtype=np.float32)*.1 for _ in range(4)]
            with patch.object(train_remover, 'ROOT', root), \
                 patch.object(train_remover, 'manifest', return_value=data), \
                 patch.object(train_remover, 'load_audio', return_value=audio), \
                 patch.object(train_remover, 'read_teacher_config', return_value={}), \
                 patch.object(train_remover, 'Teacher', DummyTeacher), \
                 patch.object(train_remover, 'validate', side_effect=[(initial, []), (damaged, [])]), \
                 patch.object(sys, 'argv', ['train_remover', '--steps', '1', '--batch-size', '1',
                     '--train-clips', '4', '--val-clips', '4', '--width', '4', '--threads', '2',
                     '--pesq-weight', '1', '--quality-pesq-floor', '4.5', '--tag', 'baseline']), \
                 contextlib.redirect_stdout(io.StringIO()):
                train_remover.main()
            best = torch.load(root/'results/remover_baseline.pt', weights_only=True)
            last = torch.load(root/'results/remover_baseline.last.pt', weights_only=True)
            self.assertEqual(best['step'], 0)
            self.assertEqual(last['step'], 1)
            self.assertEqual(best['validation'], initial)
            self.assertEqual(last['validation'], damaged)

    def test_silence_and_disabled_model_are_exact_bypasses(self):
        with torch.no_grad():
            self.model.net[-1].bias.fill_(0.7)
        for x in (np.zeros((40, 2)), np.ones((20, 2)) * 0.1):
            y = process_audio(self.model, x, 44100, strength=0)
            np.testing.assert_array_equal(x, y)
        z = np.zeros((4000, 2))
        np.testing.assert_array_equal(z, process_audio(self.model, z, 48000))

    def test_nonzero_edit_preserves_length_channels_and_high_band(self):
        with torch.no_grad():
            self.model.net[-1].bias[0] = 0.8
        sr = 44100
        t = np.arange(sr * 3 + 7) / sr
        x = np.stack([0.1 * np.sin(2 * np.pi * 2000 * t) +
                      0.1 * np.sin(2 * np.pi * 10000 * t),
                      np.zeros_like(t)], -1)
        y = process_audio(self.model, x, sr, chunk_seconds=2)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y).all())
        edit_power = np.mean((y[:, 0] - x[:, 0]) ** 2)
        self.assertGreater(edit_power, 1e-6)
        self.assertGreaterEqual(10 * np.log10(np.mean(x[:, 0] ** 2) / edit_power), 30 - 1e-8)
        np.testing.assert_array_equal(y[:, 1], x[:, 1])
        # Editing a low band must preserve the unrelated 10 kHz component.
        carrier = np.exp(-2j * np.pi * 10000 * t)
        change = abs(np.mean((y[:, 0] - x[:, 0]) * carrier))
        self.assertLess(change, 1e-5)
        # Crossfades must not introduce a seam with a large step.
        for seam in (int(sr * 1.5), int(sr * 2)):
            delta = y[:, 0] - x[:, 0]
            self.assertLess(abs(delta[seam] - delta[seam - 1]), 0.1)

    def test_very_short_inputs_at_multiple_sample_rates(self):
        with torch.no_grad():
            self.model.net[-1].bias.fill_(0.3)
        for sr in (8000, 16000, 22050, 44100, 48000):
            for shape in ((1,), (7, 2)):
                x = np.full(shape, 0.05)
                y = process_audio(self.model, x, sr)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(np.isfinite(y).all())

    def test_waveform_architecture_learns_and_enforces_the_budget(self):
        model = Remover(architecture="waveform")
        x = (torch.randn(2, 16001) * 0.1).requires_grad_()
        torch.testing.assert_close(model(x), x, rtol=0, atol=0)
        model(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(model.net[-1].weight.grad.abs().max().item(), 0)
        with torch.no_grad():
            model.net[-1].weight.normal_(0, 2)
        y = model(x)
        self.assertEqual(y.shape, x.shape)
        relative_power = (y-x).square().mean(-1) / x.square().mean(-1)
        self.assertTrue((relative_power <= 0.001001).all())
        half = model(x, strength=0.5)
        torch.testing.assert_close(half-x, 0.5*(y-x), atol=1e-7, rtol=1e-4)

    def test_invalid_audio_and_strength(self):
        for x, sr, strength in ((np.array([np.nan]), 16000, 1),
                                (np.zeros((2, 2, 2)), 16000, 1),
                                (np.ones(3), 7000, 1),
                                (np.ones(3), 16000, -1),
                                (np.ones(3), 16000, float("nan"))):
            with self.assertRaises(ValueError):
                process_audio(self.model, x, sr, strength)

    def test_local_budget_protects_quiet_speech_and_has_finite_gradients(self):
        model = Remover(local_snr_db=30)
        x = torch.randn(1, 8000) * 0.1
        x[:, 3000:5000] = 0
        x[:, 5000:] *= 0.01
        x.requires_grad_()
        delta = (torch.randn_like(x) * 0.02).requires_grad_()
        y = model._project(x, delta)
        carrier = torch.nn.functional.avg_pool1d(x[:, None].square(), 321, 1, 160)
        change = torch.nn.functional.avg_pool1d((y-x)[:, None].square(), 321, 1, 160)
        self.assertTrue((change <= carrier * 0.001001 + 1e-20).all())
        self.assertLess(float(y[:, 3500:4500].detach().abs().max()), 1e-8)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_spectral_guard_rejects_noise_away_from_speech_and_keeps_gradients(self):
        model = Remover(architecture='waveform', width=4, edit_low_hz=750,
                        edit_high_hz=7500, masking_margin_db=25, local_snr_db=30)
        t = torch.arange(16000) / 16000
        x = (.1 * torch.sin(2 * torch.pi * 1000 * t))[None].requires_grad_()
        delta = (.01 * torch.sin(2 * torch.pi * 4000 * t))[None].requires_grad_()
        y = model._project(x, delta)
        # A strong vowel cannot fund a new tone in an otherwise empty band.
        self.assertLess(float((y-x).detach().square().mean().sqrt()), 3e-5)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(delta.grad).all())
        model.zero_grad()
        x = (torch.randn(1, 8192) * .05).requires_grad_()
        torch.testing.assert_close(model(x), x, atol=0, rtol=0)
        model(x).square().mean().backward()
        self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())
        self.assertGreater(float(model.net[-1].weight.grad.abs().max()), 0)

    def test_native_metric_detects_lost_detail_above_pesq_bandwidth(self):
        t = np.arange(44100) / 44100
        body = .1 * np.sin(2*np.pi*1000*t)
        detail = .05 * np.sin(2*np.pi*10000*t)
        x = body + detail
        metrics = native_spectral_metrics(x, body + detail * .1, 44100)
        upper = next(b for b in metrics['native_bands'] if b['low_hz'] == 8000)
        self.assertAlmostEqual(upper['gain_change_db'], -20, places=2)
        self.assertGreater(metrics['native_active_bin_rms_change_db'], 10)
        self.assertEqual(native_spectral_metrics(x, x, 44100)['native_active_bin_rms_change_db'], 0)

    def test_quality_floor_precedes_detector_objective(self):
        from train_remover import selection_key
        good = dict(min_proxy_pesq=4.5, min_proxy_estoi=.995, objective=100)
        damaged = dict(min_proxy_pesq=4.3, min_proxy_estoi=.98, objective=0)
        self.assertLess(selection_key(good, 4.4, .99), selection_key(damaged, 4.4, .99))
        self.assertLess(selection_key(dict(good, objective=1), 4.4, .99), selection_key(good, 4.4, .99))
        self.assertLess(selection_key(good, 4.4, .99), selection_key(dict(good, min_proxy_pesq=float('nan')), 4.4, .99))

    def test_new_high_band_starts_from_identical_audio_and_receives_gradients(self):
        base = Remover(architecture='waveform', width=8, context_layers=1, local_snr_db=27)
        with torch.no_grad():
            base.net[-1].weight.normal_(0, .1)
        model = Remover(**dict(base.config, extra_low_hz=4000, extra_high_hz=7500))
        missing = model.load_state_dict(base.state_dict(), strict=False).missing_keys
        self.assertEqual(set(missing), {'extra_head.weight', 'extra_head.bias'})
        x = (torch.randn(1, 16000) * .05).requires_grad_()
        y = model(x)
        torch.testing.assert_close(y, base(x), rtol=0, atol=0)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(model.extra_head.weight.grad).all())
        self.assertGreater(float(model.extra_head.weight.grad.abs().max()), 0)

    def test_high_band_edits_follow_high_band_speech_energy(self):
        model = Remover(architecture='waveform', width=8, local_snr_db=27,
                        extra_low_hz=4000, extra_high_hz=7500)
        with torch.no_grad():
            model.extra_head.weight.normal_(0, .5)
        t = torch.arange(16000) / 16000
        x = .1 * torch.sin(2 * torch.pi * 2000 * t)
        x[8000:] += .1 * torch.sin(2 * torch.pi * 6000 * t[8000:])
        x = x[None].requires_grad_()
        y = model(x)
        delta = y - x
        quiet = delta[:, 1600:6400].square().mean().sqrt()
        active = delta[:, 9600:14400].square().mean().sqrt()
        self.assertLess(float(quiet.detach()), float(active.detach()) * .01)
        self.assertLess(float(delta.detach().square().mean() / x.detach().square().mean()), .001001)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_phase_only_edit_preserves_tonal_magnitudes_despite_waveform_difference(self):
        model = Remover(width=4, max_phase=3, max_gain=0, min_snr_db=0)
        with torch.no_grad():
            model.net[-1].bias[0] = float(np.arctanh(1/3))
        rate = 48000
        t = np.arange(rate) / rate
        x = .1 * (np.sin(2*np.pi*300*t) + np.sin(2*np.pi*2000*t) + np.sin(2*np.pi*10000*t))
        y = process_audio(model, x, rate)
        region = slice(4800, -4800)
        for frequency in (300, 2000, 10000):
            carrier = np.exp(-2j*np.pi*frequency*t[region])
            amplitude = 2 * abs(np.mean(y[region] * carrier))
            # Finite resampling filters add a small passband ripple to the
            # resampled residual; allow 0.2% amplitude error at native rate.
            self.assertAlmostEqual(amplitude, .1, delta=.0002)
        self.assertGreater(np.sqrt(np.mean((y-x)**2)), .05)
        self.assertTrue(np.isfinite(y).all())

    def test_sharper_residual_resampling_rejects_high_frequency_images(self):
        t16 = np.arange(16000) / 16000
        x = .1 * np.sin(2*np.pi*7500*t16)
        ordinary = resample(x, 16000, 48000)
        sharp = resample(x, 16000, 48000, filter_half_length=64)
        t48 = np.arange(48000) / 48000
        region = slice(4800, -4800)
        def amplitude(y, frequency):
            return 2*abs(np.mean(y[region]*np.exp(-2j*np.pi*frequency*t48[region])))
        self.assertLess(amplitude(sharp, 8500), 1e-5)
        self.assertGreater(amplitude(ordinary, 8500), amplitude(sharp, 8500)*100)
        self.assertAlmostEqual(amplitude(sharp, 7500), .1, delta=.0001)

    def test_native_band_replacement_cannot_copy_original_band_details(self):
        from rebuild_hf import replace_native_band
        rate = 44100
        t = np.arange(rate) / rate
        outside = .1*np.sin(2*np.pi*1000*t) + .03*np.sin(2*np.pi*10000*t)
        a = outside + .1*np.sin(2*np.pi*7000*t)
        b = outside + .1*np.sin(2*np.pi*7300*t)
        prediction = .08*np.sin(2*np.pi*7500*t)
        y1, _ = replace_native_band(a, prediction, rate)
        y2, _ = replace_native_band(b, prediction, rate)
        np.testing.assert_allclose(y1, y2, atol=1e-10, rtol=0)
        expected = outside + .1*np.sin(2*np.pi*7500*t)
        np.testing.assert_allclose(y1, expected, atol=1e-10, rtol=0)

    def test_added_context_starts_from_the_same_audio_and_receives_gradients(self):
        baseline = Remover(architecture="waveform", width=8, local_snr_db=27)
        with torch.no_grad():
            baseline.net[-1].weight.normal_(0, 0.02)
        expanded = Remover(architecture="waveform", width=8, local_snr_db=27, context_layers=3)
        missing = expanded.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(all(k.startswith("context.") for k in missing.missing_keys))
        x = torch.randn(1, 5000) * 0.1
        torch.testing.assert_close(baseline(x), expanded(x), rtol=0, atol=0)
        expanded(x).square().mean().backward()
        self.assertTrue(torch.isfinite(expanded.context[0].weight.grad).all())
        self.assertGreater(expanded.context[0].weight.grad.abs().max().item(), 0)

    def test_differentiable_intelligibility_loss_matches_identity_and_backpropagates(self):
        from third_party.stoi_loss import NegSTOILoss
        metric = NegSTOILoss(16000, extended=True, do_resample=False)
        x = torch.randn(1, 32000) * 0.1
        self.assertGreater(float(-metric(x, x)[0]), 0.999)
        y = (x + torch.randn_like(x) * 0.01).requires_grad_()
        loss = metric(y, x).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(y.grad).all())
        self.assertGreater(y.grad.abs().max().item(), 0)

    def test_perceptual_filter_matches_scipy_and_quality_loss_has_gradients(self):
        from third_party.torch_pesq.ops import lfilter
        from third_party.torch_pesq import PesqLoss
        from scipy.signal import lfilter as reference_filter
        x = torch.randn(1, 32000) * 0.1
        a, b = torch.tensor([1.0, -0.95]), torch.tensor([0.05, 0.0])
        y = lfilter(x, a, b)
        reference = reference_filter(b.numpy(), a.numpy(), x.numpy(), axis=-1)
        np.testing.assert_allclose(y.numpy(), reference, atol=1e-6, rtol=1e-4)
        quality = PesqLoss(1, sample_rate=16000)
        z = (x + 0.01 * torch.randn_like(x)).requires_grad_()
        score = quality.mos(x, z)
        self.assertLess(float(score.detach()[0]), float(quality.mos(x, x)[0]))
        quality(x, z).sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_api_cooldown_cancels_a_waiting_request(self):
        import cartesia
        with tempfile.TemporaryDirectory() as directory:
            clock = [100.0]
            def sleep_then_cooldown(seconds):
                cartesia._pace_detector(cooldown=True)
                clock[0] += seconds
            with patch.object(cartesia, "ROOT", Path(directory)), \
                 patch.object(cartesia, "MIN_INTERVAL", 5), \
                 patch.object(cartesia.time, "time", side_effect=lambda: clock[0]), \
                 patch.object(cartesia.time, "sleep", side_effect=sleep_then_cooldown):
                cartesia._pace_detector()
                with self.assertRaises(cartesia.RateLimited):
                    cartesia._pace_detector()
                self.assertEqual(clock[0], 105.0)

    def test_band_reconstruction_cannot_copy_the_removed_input_band(self):
        from band_rebuilder import BandRebuilder
        model=BandRebuilder(width=4)
        with torch.no_grad():
            model.last.weight.normal_(0,.02)
        x=torch.randn(1,8192)*.05
        tone=torch.tensor(.1*np.sin(2*np.pi*2000*np.arange(8192)/16000),dtype=torch.float32)[None]
        # Two inputs differing only inside the removed band must reconstruct
        # the same output, including when the network predicts nonzero speech.
        torch.testing.assert_close(model(x),model(x+tone),atol=2e-6,rtol=1e-5)
        model(x).square().mean().backward()
        self.assertTrue(torch.isfinite(model.first.weight.grad).all())

    def test_exported_band_model_preserves_native_shape_and_energy_budget(self):
        from band_rebuilder import BandRebuilder
        model=BandRebuilder(width=4,mix=.75,min_snr_db=22)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'band.pt'
            torch.save(dict(format='speech-band-rebuilder-v1',model_config=model.config,state_dict=model.state_dict()),checkpoint)
            restored,_=load_model(checkpoint)
            x=np.random.default_rng(8).normal(0,.05,(12000,2))
            y=process_audio(restored,x,24000)
            self.assertEqual(x.shape,y.shape)
            self.assertGreaterEqual(quality_metrics(x,y)['snr_db'],21.999)
            self.assertTrue(np.isfinite(y).all())
            np.testing.assert_array_equal(process_audio(restored,x,24000,strength=0),x)

    def test_fft_reconstruction_resampling_preserves_tone_and_input_gradients(self):
        from codec_teacher import fft_resample
        x=torch.tensor(np.sin(2*np.pi*2000*np.arange(16000)/16000),dtype=torch.float32)[None].requires_grad_()
        y=fft_resample(fft_resample(x,24000),16000)
        torch.testing.assert_close(x,y,atol=2e-6,rtol=1e-5)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_native_filter_reaches_above_neural_band_and_strength_zero_bypasses(self):
        model=Remover(architecture='waveform',width=4,post_lowpass_hz=6000)
        t=np.arange(48000)/48000
        low=np.sin(2*np.pi*2000*t);high=np.sin(2*np.pi*10000*t)
        x=.1*low+.05*high
        y=process_audio(model,x,48000)
        region=slice(4800,-4800)
        self.assertAlmostEqual(float(2*np.mean(y[region]*low[region])),.1,places=5)
        self.assertLess(abs(float(2*np.mean(y[region]*high[region]))),.0001)
        np.testing.assert_array_equal(process_audio(model,x,48000,strength=0),x)
        np.testing.assert_allclose(process_audio(model,x,48000,strength=.5),(x+y)/2,atol=1e-12)
        for n in (1,8,32):
            short=process_audio(model,x[:n],48000)
            self.assertEqual(short.shape,(n,))
            self.assertTrue(np.isfinite(short).all())

    def test_noise_excited_phase_repair_is_repeatable_and_finite(self):
        from band_rebuilder import BandRebuilder
        model=BandRebuilder(width=4,low=4000,high=7900,noise_input=True,phase_hint=True,
                            magnitude_blur=2,magnitude_time_blur=1).eval()
        x=torch.randn(1,4096)*.05
        a,b=model(x),model(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(a.shape,x.shape)
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(model(torch.zeros_like(x)),torch.zeros_like(x),rtol=0,atol=1e-12)

    def test_ensemble_teacher_carries_nested_voice_exclusions_and_rejects_cycles(self):
        import json
        from train_remover import read_teacher_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child, middle, ensemble = [root/name for name in ('child.json', 'middle.json', 'ensemble.json')]
            child.write_text(json.dumps({'training_voices': ['seen'], 'calibration_voices': ['calibrated']}))
            middle.write_text(json.dumps({'ensemble_configs': [str(child)]}))
            ensemble.write_text(json.dumps({'ensemble_configs': [str(middle)], 'training_voices': ['parent']}))
            config = read_teacher_config(ensemble)
            self.assertEqual(set(config['training_voices']), {'seen', 'parent'})
            self.assertEqual(config['calibration_voices'], ['calibrated'])
            child.write_text(json.dumps({'ensemble_configs': [str(ensemble)]}))
            with self.assertRaisesRegex(ValueError, 'Cyclic'):
                read_teacher_config(ensemble)

    def test_residual_scale_matches_strength_and_preserves_budgets(self):
        source = Remover(width=4, architecture='waveform', local_snr_db=27).eval()
        scaled = Remover(**dict(source.config, residual_scale=.95)).eval()
        with torch.no_grad():
            source.net[-1].weight.normal_(0, .02)
        scaled.load_state_dict(source.state_dict())
        x = torch.randn(2, 4096)*.05
        torch.testing.assert_close(scaled(x), source(x, strength=.95), rtol=0, atol=0)
        torch.testing.assert_close(scaled(x, strength=0), x, rtol=0, atol=0)

    def test_quality_blend_enforces_absolute_floor_and_falls_back_to_baseline(self):
        from quality_blend import QualityBlend
        first = Remover(width=4, architecture='waveform', min_snr_db=20).eval()
        baseline = Remover(width=4, architecture='waveform').eval()
        with torch.no_grad():
            first.net[-1].weight.normal_(0, .05)
        model = QualityBlend([first.config, baseline.config]).eval()
        for component, source in zip(model.components, (first, baseline)):
            component.load_state_dict(source.state_dict())
        x = np.random.default_rng(8).normal(0, .05, 32000)
        with patch.object(model, '_metrics', side_effect=[np.array([[4.52, .997]]),
                np.array([[4.495, .9965]]), np.array([[4.515, .9965]])]):
            y = process_audio(model, x, 16000)
        self.assertEqual(model.last_quality_decision['fraction'], .125)
        self.assertTrue(np.isfinite(y).all())
        self.assertGreater(np.max(np.abs(y-x)), 1e-8)
        with patch.object(model, '_metrics', side_effect=[np.array([[4.52, .997]]),
                np.array([[4.48, .996]]), np.array([[4.49, .996]])]):
            y = process_audio(model, x, 16000)
        np.testing.assert_array_equal(y, x.astype(np.float32).astype(np.float64))
        self.assertEqual(model.last_quality_decision['fraction'], 0)
        np.testing.assert_array_equal(process_audio(model, np.zeros(8), 16000), np.zeros(8))
        self.assertIsNone(model.last_quality_decision)

    def test_oracle_pairs_require_same_utterance_not_just_same_voice(self):
        from train_native_oracle import same_source_groups, paired_metrics
        records = [
            {'voice': 'voice', 'label': True, 'reference_sources': ['a'], 'kind': 'natural', 'logit': 1.},
            {'voice': 'voice', 'label': False, 'reference_sources': ['a'], 'kind': 'high_quality_edit', 'logit': -1.},
            {'voice': 'voice', 'label': True, 'reference_sources': ['b'], 'kind': 'natural', 'logit': -1.},
            {'voice': 'voice', 'label': False, 'reference_sources': ['c'], 'kind': 'high_quality_edit', 'logit': 1.},
        ]
        self.assertEqual(len(same_source_groups(records)), 1)
        metric = paired_metrics(records, high_quality=True)
        self.assertEqual((metric['utterances'], metric['pairs'], metric['accuracy']), (1, 1, 1.))
        self.assertAlmostEqual(metric['loss'], float(np.logaddexp(0, -1.5)))
        records[1]['voice'] = 'different'
        with self.assertRaisesRegex(ValueError, 'different voices'):
            same_source_groups(records)

    def test_differentiable_complete_file_render_matches_inference_across_overlap(self):
        from remover import render_chunks16k
        model = Remover(width=4, architecture='waveform', min_snr_db=30, local_snr_db=27).eval()
        with torch.no_grad():
            model.net[-1].weight.normal_(0, .03)
        x = torch.from_numpy(np.random.default_rng(9).normal(0, .05, (2, 103200)).astype(np.float32))
        rendered = render_chunks16k(model, x)
        expected = process_audio(model, x.numpy().T.astype(np.float64), 16000).T.astype(np.float32)
        np.testing.assert_allclose(rendered.detach().numpy(), expected, rtol=0, atol=6e-8)
        rendered.square().mean().backward()
        self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())
        self.assertGreater(float(model.net[-1].weight.grad.abs().sum()), 0.)

    def test_stored_detector_features_keep_tiny_float32_gradients(self):
        from train_remover import stored_frame_values
        x = torch.randn(2, 150, 16, requires_grad=True)
        quantized = stored_frame_values(x)
        torch.testing.assert_close(quantized, x.detach().half().float(), rtol=0, atol=0)
        (quantized*1e-9).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 1e-9), rtol=0, atol=0)

    def test_suppression_control_skip_preserves_loss_and_input_gradients(self):
        from train_remover import suppression_loss
        class Target:
            margin, cross_encoder = .5, None
            def __call__(self, x):
                return x.mean(-1)
        teacher = Target()
        x = torch.randn(3, 128, requires_grad=True)
        marked = torch.tensor([True, False, True])
        expected = torch.nn.functional.softplus(teacher(x)+.5)*marked
        actual = suppression_loss(teacher, x, marked)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        a, = torch.autograd.grad(actual.sum(), x, retain_graph=True)
        b, = torch.autograd.grad(expected.sum(), x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_detector_http_calls_are_serial_even_when_start_pacing_is_disabled(self):
        import cartesia
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock
        import time
        from types import SimpleNamespace
        active, peak = [0], [0]
        guard = Lock()
        def post(*args, **kwargs):
            with guard:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(.02)
            with guard:
                active[0] -= 1
            return SimpleNamespace(status_code=200, json=lambda: {'detected': True})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'input.wav'; source.write_bytes(b'test')
            with patch.object(cartesia, 'ROOT', Path(directory)), \
                 patch.object(cartesia, '_env', return_value=None), \
                 patch.object(cartesia, '_pace_detector'), \
                 patch.object(cartesia.requests, 'post', side_effect=post), \
                 ThreadPoolExecutor(max_workers=3) as pool:
                pending = [pool.submit(cartesia.detect, source) for _ in range(3)]
                for future in pending:
                    self.assertEqual(future.result(timeout=2), (200, {'detected': True}))
            self.assertEqual(peak[0], 1)

    def test_checkpoint_and_cli_roundtrip_and_source_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, source, output = root / "model.pt", root / "source.wav", root / "output.wav"
            torch.save({"format": "cartesia-remover-v1", "model_config": self.model.config,
                        "state_dict": self.model.state_dict()}, checkpoint)
            restored, _ = load_model(checkpoint)
            np.testing.assert_array_equal(process_audio(restored, np.ones(5), 16000), np.ones(5))
            sf.write(source, np.ones((600, 2)) * 0.125, 22050, subtype="FLOAT")
            original = source.read_bytes()
            with patch.object(sys, "argv", ["remove_watermark", str(source), str(output),
                                              "--checkpoint", str(checkpoint)]), contextlib.redirect_stdout(io.StringIO()):
                remove_watermark.main()
            audio, sr = sf.read(output)
            self.assertEqual((sr, audio.shape), (22050, (600, 2)))
            self.assertEqual(source.read_bytes(), original)
            alias = root / "alias.wav"
            alias.hardlink_to(source)
            for target in (source, output, alias):
                with patch.object(sys, "argv", ["remove_watermark", str(source), str(target)]), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    remove_watermark.main()
            self.assertEqual(source.read_bytes(), original)

    def test_official_verification_pairs_labels_and_reuses_only_matching_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            checkpoint, source = root / "model.pt", root / "source.wav"
            checkpoint.write_bytes(b"checkpoint")
            x = 0.1 * np.sin(2 * np.pi * 2000 * np.arange(83200) / 16000)
            sf.write(source, x, 16000, subtype="FLOAT")
            with torch.no_grad():
                self.model.net[-1].bias[0] = 0.7
            with patch.object(verify_remover, "ROOT", root), \
                 patch.object(verify_remover, "load_model", return_value=(self.model, {})), \
                 patch.object(verify_remover.cartesia, "_env", return_value="test"), \
                 patch.object(verify_remover.cartesia, "detect", side_effect=[
                     (200, {"detected": True}), (200, {"detected": False}),
                     (200, {"detected": False})]) as detect, \
                 contextlib.redirect_stdout(io.StringIO()):
                for tag in ("first", "second"):
                    with patch.object(sys, "argv", ["verify_remover", "--checkpoint", str(checkpoint),
                                                     "--tag", tag, str(source)]):
                        self.assertEqual(verify_remover.main(), 0)
                # FLOAT WAV headers can contain a fresh PEAK timestamp, so a
                # newly encoded output may require a fresh hash-bound query.
                self.assertIn(detect.call_count, (2, 3))
                self.assertEqual(sum(call.args[0] == source for call in detect.call_args_list), 1)
            import json
            report = json.loads((root / "results/remover_second_cartesia.json").read_text())
            self.assertEqual(report["summary"]["original_positive_edited_negative"], 1)
            self.assertEqual(report["summary"]["edited_detected"], 0)


if __name__ == "__main__":
    unittest.main()
