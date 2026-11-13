"""Small, trainable audio editors and sample-rate-preserving inference.

The detector-guided checkpoint suppresses a detector; that is not proof of
recovering the waveform before watermarking. Inference needs no detector/API.
"""
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import butter, firwin, resample_poly, sosfiltfilt, stft
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
DEFAULT_CHECKPOINT = ROOT / "results/remover_quality.pt"


class Remover(nn.Module):
    def __init__(self, width=16, max_phase=0.5, max_gain=0.15, min_snr_db=30.0, architecture="spectral",
                 local_snr_db=None, context_layers=0, waveform_gain=0.1, post_lowpass_hz=None,
                 edit_low_hz=1000, edit_high_hz=4000, masking_margin_db=None,
                 extra_low_hz=None, extra_high_hz=None, extra_gain=0.3,
                 residual_filter_half_length=None, residual_scale=1.0):
        super().__init__()
        if not 0 < edit_low_hz < edit_high_hz < SAMPLE_RATE / 2:
            raise ValueError("Edit frequencies must lie between zero and Nyquist")
        if masking_margin_db is not None and masking_margin_db < 0:
            raise ValueError("Spectral masking margin must be nonnegative")
        if (extra_low_hz is None) != (extra_high_hz is None):
            raise ValueError("Specify both extra-band edges")
        if extra_low_hz is not None and (architecture != 'waveform' or
                not 0 < extra_low_hz < extra_high_hz < SAMPLE_RATE / 2):
            raise ValueError("Extra band requires waveform architecture and valid frequencies")
        if not 0 < extra_gain <= 2:
            raise ValueError("Extra-band gain must be in (0, 2]")
        if not np.isfinite(residual_scale) or not 0 < residual_scale <= 1:
            raise ValueError('Residual scale must be in (0, 1]')
        if residual_filter_half_length is not None and (not isinstance(residual_filter_half_length, int) or
                                                       not 1 <= residual_filter_half_length <= 256):
            raise ValueError('Residual filter half length must be an integer in [1, 256]')
        self.config = dict(width=width, max_phase=max_phase, max_gain=max_gain,
                           min_snr_db=min_snr_db, architecture=architecture, local_snr_db=local_snr_db,
                           context_layers=context_layers, waveform_gain=waveform_gain,
                           post_lowpass_hz=post_lowpass_hz, edit_low_hz=edit_low_hz,
                           edit_high_hz=edit_high_hz, masking_margin_db=masking_margin_db,
                           extra_low_hz=extra_low_hz, extra_high_hz=extra_high_hz, extra_gain=extra_gain,
                           residual_filter_half_length=residual_filter_half_length, residual_scale=residual_scale)
        if architecture == "spectral":
            self.net = nn.Sequential(
                nn.Conv2d(4, width, 3, padding=1), nn.SiLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.SiLU(),
                nn.Conv2d(width, 2, 1),
            )
        elif architecture == "waveform":
            self.net = nn.Sequential(
                nn.Conv1d(1, width, 15, stride=2, padding=7), nn.SiLU(),
                nn.Conv1d(width, width, 7, padding=6, dilation=2), nn.SiLU(),
                nn.Conv1d(width, width, 7, padding=12, dilation=4), nn.SiLU(),
                nn.Conv1d(width, width, 7, padding=24, dilation=8), nn.SiLU(),
                nn.ConvTranspose1d(width, 1, 16, stride=2, padding=7),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.context = nn.ModuleList()
        if architecture == "waveform":
            for i in range(context_layers):
                dilation = 16 * 2 ** i
                layer = nn.Conv1d(width, width, 7, padding=3 * dilation, dilation=dilation)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                self.context.append(layer)
        self.register_buffer("window", torch.hann_window(512), persistent=False)
        freq = torch.linspace(0, SAMPLE_RATE / 2, 257)
        # Defaults retain the original checkpoint's spectral shoulders.
        band = ((freq - edit_low_hz + 250) / 500).clamp(0, 1) * ((edit_high_hz + 500 - freq) / 500).clamp(0, 1)
        self.register_buffer("band", band[None, :, None], persistent=False)
        fir = torch.tensor(firwin(129, [edit_low_hz, edit_high_hz], pass_zero=False, fs=SAMPLE_RATE), dtype=torch.float32)
        self.register_buffer("bandpass_fir", fir[None, None], persistent=False)
        self.extra_head = None
        if extra_low_hz is not None:
            # Open a new band without revealing untrained high-frequency output
            # from the existing head. The initial waveform stays exactly intact.
            self.extra_head = nn.ConvTranspose1d(width, 1, 16, stride=2, padding=7)
            nn.init.zeros_(self.extra_head.weight)
            nn.init.zeros_(self.extra_head.bias)
            extra_fir = torch.tensor(firwin(129, [extra_low_hz, extra_high_hz],
                                           pass_zero=False, fs=SAMPLE_RATE), dtype=torch.float32)
            self.register_buffer('extra_fir', extra_fir[None, None], persistent=False)

    def forward(self, x, strength=1.0):
        """B x samples at 16 kHz -> same shape, with differentiable bounded edits."""
        strength = strength * self.config['residual_scale']
        n = x.shape[-1]
        scale = x.square().mean(-1, keepdim=True).clamp_min(1e-10).sqrt()
        if self.config["architecture"] == "waveform":
            hidden = self.net[:-1]((x / scale)[:, None])
            for layer in self.context:
                hidden = hidden + layer(nn.functional.silu(hidden))
            raw = self.net[-1](hidden)[:, :, :n].tanh()
            filtered = nn.functional.conv1d(raw, self.bandpass_fir, padding=64)[:, 0]
            delta = filtered * scale * self.config['waveform_gain']
            if self.extra_head is not None:
                carrier = nn.functional.conv1d(x[:, None], self.extra_fir, padding=64)
                envelope = nn.functional.avg_pool1d(carrier.square(), 321, 1, 160).clamp_min(1e-20).sqrt()
                extra = self.extra_head(hidden)[:, :, :n].tanh() * envelope * self.config['extra_gain']
                delta = delta + nn.functional.conv1d(extra, self.extra_fir, padding=64)[:, 0]
            return self._project(x, delta * strength, strength)
        z = torch.stft(x / scale, 512, 128, window=self.window,
                       return_complex=True, pad_mode="constant")
        mag = z.abs().clamp_min(1e-7)
        freq = torch.linspace(0, 1, z.shape[1], device=x.device)[None, :, None].expand_as(mag)
        features = torch.stack((torch.log1p(mag), z.real / mag, z.imag / mag, freq), 1)
        mask = self.net(features).tanh() * self.band[:, None] * strength
        phase = mask[:, 0] * self.config["max_phase"]
        gain = mask[:, 1] * self.config["max_gain"]
        edited = z * torch.polar(1 + gain, phase)
        y = torch.istft(edited, 512, 128, window=self.window, length=n) * scale
        # Exact bypass at strength=0 and for the zero-initialized network.
        baseline = torch.istft(z, 512, 128, window=self.window, length=n) * scale
        return self._project(x, y - baseline, strength)

    def _project(self, x, delta, strength=1.0):
        if self.config["masking_margin_db"] is not None:
            # Bound edits by nearby speech energy, rather than allowing a loud
            # vowel to fund noise in an unrelated frequency band. This is a
            # spectral proxy, not a calibrated audibility threshold. Reanalysis
            # after overlap-add is approximate; time-domain bounds below remain
            # strict. Two passes reduce STFT consistency leakage into quiet bins.
            reference = torch.stft(x, 512, 128, window=self.window,
                                   return_complex=True, pad_mode="constant")
            envelope = nn.functional.avg_pool2d(reference.abs().square()[:, None],
                                                (9, 3), stride=1, padding=(4, 1))[:, 0]
            limit = (envelope.clamp_min(1e-24).sqrt() *
                     10 ** (-self.config["masking_margin_db"] / 20) * min(strength, 1.0))
            for _ in range(2):
                spectrum = torch.stft(delta, 512, 128, window=self.window,
                                      return_complex=True, pad_mode="constant")
                factor = (limit / spectrum.abs().clamp_min(1e-12)).clamp(max=1)
                delta = torch.istft(spectrum * factor, 512, 128, window=self.window, length=x.shape[-1])
        if self.config["local_snr_db"] is not None:
            # Every overlapping 20 ms window constrains the samples it covers.
            # Taking the minimum neighbouring gain prevents a quiet consonant
            # from inheriting the budget of a louder vowel elsewhere in the clip.
            carrier = nn.functional.avg_pool1d(x[:, None].square(), 321, 1, 160)
            change = nn.functional.avg_pool1d(delta[:, None].square(), 321, 1, 160)
            ratio = (carrier.clamp_min(1e-20) / change.clamp_min(1e-24)).sqrt()
            ratio = (ratio * 10 ** (-self.config["local_snr_db"] / 20) * min(strength, 1)).clamp(max=1)
            factor = -nn.functional.max_pool1d(-ratio, 321, 1, 160)
            delta = delta * factor[:, 0]
        budget = (x.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt() *
                  10 ** (-self.config["min_snr_db"] / 20) * min(strength, 1.0))
        factor = (budget / delta.square().mean(-1, keepdim=True).clamp_min(1e-24).sqrt()).clamp(max=1)
        return x + delta * factor


def load_model(path=DEFAULT_CHECKPOINT, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('format') == 'speech-band-rebuilder-v1':
        from band_rebuilder import BandRebuilder
        model = BandRebuilder(**checkpoint['model_config']).to(device)
    elif checkpoint.get('format') == 'cartesia-quality-blend-v1':
        from quality_blend import QualityBlend
        model = QualityBlend(**checkpoint['model_config']).to(device)
    elif checkpoint.get("format") == "cartesia-remover-v1":
        model = Remover(**checkpoint["model_config"]).to(device)
    else:
        raise ValueError("Unsupported remover checkpoint format")
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def resample(x, old_rate, new_rate, filter_half_length=None):
    if old_rate == new_rate:
        return x.copy()
    g = gcd(int(old_rate), int(new_rate))
    up, down = new_rate // g, old_rate // g
    if filter_half_length is None:
        return resample_poly(x, up, down, axis=0)
    # A wider-band editor needs a sharper reconstruction filter to keep its
    # near-Nyquist edits from creating images above the 16 kHz path's bandwidth.
    factor = max(up, down)
    window = firwin(2*filter_half_length*factor+1, 1/factor, window=('kaiser', 8.6))
    return resample_poly(x, up, down, axis=0, window=window)


def render_chunks16k(model, audio, strength=1., chunk_seconds=6.):
    """Differentiable version of inference chunking for complete-file training.

    B x samples at 16 kHz. Keep the per-chunk normalization, overlap weights,
    short final chunk, and final whole-recording energy constraint used below.
    """
    n = audio.shape[-1]
    if not n or strength == 0:
        return audio
    chunk = int(chunk_seconds*SAMPLE_RATE)
    overlap = min(SAMPLE_RATE, chunk//4)
    accum = torch.zeros_like(audio, dtype=torch.float64)
    weights = torch.zeros(n, device=audio.device, dtype=torch.float64)
    for start in range(0, n, chunk-overlap):
        end = min(n, start+chunk)
        x = audio[:, start:end]
        residual = (model(x, strength)-x).to(torch.float64)
        weight = torch.ones(end-start, device=audio.device, dtype=torch.float64)
        if start:
            fade = min(overlap, end-start)
            weight[:fade] = torch.linspace(0, 1, fade+2, device=audio.device, dtype=torch.float64)[1:-1]
        if end < n:
            fade = min(overlap, end-start)
            weight[-fade:] = torch.linspace(1, 0, fade+2, device=audio.device, dtype=torch.float64)[1:-1]
        accum = accum + nn.functional.pad(residual*weight, (start, n-end))
        weights[start:end] += weight
        if end == n:
            break
    delta = accum/weights
    if model.config.get('min_snr_db') is not None:
        budget = (audio.to(torch.float64).square().mean(-1, keepdim=True).sqrt() *
                  10**(-model.config['min_snr_db']/20)*min(strength, 1.))
        factor = (budget/delta.square().mean(-1, keepdim=True).clamp_min(1e-40).sqrt()).clamp(max=1.)
        delta = delta*factor
    return (audio.to(torch.float64)+delta).to(audio.dtype)


@torch.inference_mode()
def process_audio(model, audio, sample_rate, strength=1.0, chunk_seconds=6.0):
    """Preserve sample rate, channel count and length.

    Only the learned residual is resampled back. Channels are processed
    independently; long clips use overlapping, cross-faded chunks.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[1] < 1):
        raise ValueError("Audio must have shape (samples,) or (samples, channels)")
    if not np.isfinite(x).all():
        raise ValueError("Audio contains non-finite samples")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate < 8000:
        raise ValueError("Sample rate must be an integer of at least 8000 Hz")
    if not np.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError("Strength must be between 0 and 2")
    if not np.isfinite(chunk_seconds) or chunk_seconds < 2:
        raise ValueError("Chunks must be at least two seconds long")
    if hasattr(model, 'last_quality_decision'):
        model.last_quality_decision = None
    if not len(x) or strength == 0 or not np.any(x):
        return x.copy()
    if hasattr(model, 'process_native'):
        return model.process_native(x, sample_rate, strength, chunk_seconds)
    mono = x.ndim == 1
    channels = x[:, None] if mono else x
    y16 = resample(channels, sample_rate, SAMPLE_RATE)
    n = len(y16)
    chunk = int(chunk_seconds * SAMPLE_RATE)
    overlap = min(SAMPLE_RATE, chunk // 4)
    hop = chunk - overlap
    accum = np.zeros_like(y16)
    weights = np.zeros(n)
    device = next(model.parameters()).device
    for start in range(0, n, hop):
        end = min(n, start + chunk)
        z = torch.as_tensor(y16[start:end].T.copy(), dtype=torch.float32, device=device)
        residual = (model(z, strength) - z).cpu().numpy().T
        weight = np.ones(end - start)
        if start:
            fade = min(overlap, end - start)
            weight[:fade] = np.linspace(0, 1, fade + 2)[1:-1]
        if end < n:
            fade = min(overlap, end - start)
            weight[-fade:] = np.linspace(1, 0, fade + 2)[1:-1]
        accum[start:end] += residual * weight[:, None]
        weights[start:end] += weight
        if end == n:
            break
    delta = resample(accum / weights[:, None], SAMPLE_RATE, sample_rate,
                     filter_half_length=model.config.get('residual_filter_half_length'))
    delta = delta[:len(x)]
    if len(delta) != len(x):
        delta = np.pad(delta, ((0, len(x) - len(delta)), (0, 0)))
    # Enforce the same energy budget after crossfades and rate conversion, per channel.
    if model.config.get('min_snr_db') is not None:
        budget = (np.sqrt(np.mean(channels ** 2, axis=0)) *
                  10 ** (-model.config["min_snr_db"] / 20) * min(strength, 1.0))
        factor = np.minimum(1, budget / np.maximum(np.sqrt(np.mean(delta ** 2, axis=0)), 1e-20))
        delta = delta * factor
    out = channels + delta
    cutoff = model.config.get('post_lowpass_hz')
    if cutoff is not None and cutoff < sample_rate/2:
        # Native-rate processing also reaches frequencies absent from the
        # 16 kHz neural training path. Do not scale it back using the neural
        # residual budget: doing so would retain the rejected frequency range.
        sos = butter(10, cutoff, 'lowpass', fs=sample_rate, output='sos')
        filtered = sosfiltfilt(sos, out, axis=0, padlen=min(33,len(out)-1))
        out = filtered if strength >= 1 else out + strength*(filtered-out)
    if not np.isfinite(out).all():
        raise RuntimeError("Model produced non-finite audio")
    return out[:, 0] if mono else out


def quality_metrics(original, edited):
    x, y = np.asarray(original, np.float64), np.asarray(edited, np.float64)
    if x.shape != y.shape:
        raise ValueError("Audio shapes differ")
    signal = float(np.mean(x ** 2))
    noise = float(np.mean((x - y) ** 2))
    return {"snr_db": float(10 * np.log10((signal + 1e-20) / (noise + 1e-20))),
            "peak_change": float(np.max(np.abs(x - y))),
            "output_peak": float(np.max(np.abs(y)))}


def native_spectral_metrics(original, edited, sample_rate):
    """Measure native-band coloration that 16 kHz PESQ cannot describe.

    Gain changes use whole-file band power. The finer spectral distance uses
    reference-active bins within 40 dB of each frame's peak. Neither measure is
    a perceptual opinion score or a guarantee of inaudibility.
    """
    x, y = np.asarray(original, np.float64), np.asarray(edited, np.float64)
    if x.shape != y.shape or not x.size:
        raise ValueError("Expected matching, nonempty audio")
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    size = min(len(x), 2 ** int(np.ceil(np.log2(sample_rate * .032))))
    frequency, _, a = stft(x, sample_rate, nperseg=size, noverlap=3 * size // 4, axis=0)
    _, _, b = stft(y, sample_rate, nperseg=size, noverlap=3 * size // 4, axis=0)
    power_a, power_b = np.abs(a) ** 2, np.abs(b) ** 2
    active = (power_a > power_a.max(axis=0, keepdims=True) * 1e-4) & (power_a > 1e-20)
    difference = 10 * np.log10(np.maximum(power_b, 1e-24) / np.maximum(power_a, 1e-24))
    distance = float(np.sqrt(np.mean(difference[active] ** 2))) if np.any(active) else 0.0
    bands = []
    total = float(power_a.sum())
    edges = (0, 500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 24000)
    for low, high in zip(edges, edges[1:]):
        high = min(high, sample_rate / 2)
        bins = (frequency >= low) & (frequency < high)
        if high <= low or not np.any(bins):
            continue
        before, after = float(power_a[bins].sum()), float(power_b[bins].sum())
        relative = 10 * np.log10((before + 1e-24) / (total + 1e-24))
        bands.append({"low_hz": low, "high_hz": high,
                      "reference_relative_power_db": float(relative),
                      "gain_change_db": float(10 * np.log10((after + 1e-24) / (before + 1e-24)))})
    active_bands = [abs(band["gain_change_db"]) for band in bands if band["reference_relative_power_db"] >= -40]
    return {"native_active_bin_rms_change_db": distance,
            "native_max_active_band_gain_change_db": max(active_bands, default=0.0),
            "native_bands": bands}
