"""Research adapter for changing a pinned codec's input frame alignment.

Circular offsets are applied at 24 kHz before lossy coding and undone before
restoring the source rate. This changes quantizer framing without deliberately
shifting the final recording. It is an untested removal hypothesis. Offsets can
also change boundary context and speech; the normal content and quality gates
remain necessary. This module performs no detector or synthesis API requests.

The active prospective study does not import this module. Real-model probes
are deferred until that study and its withheld transcription scoring finish.
"""
import hashlib
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from codec_audio import validate_audio

NATIVE_RATE = 24000
FRAME_SAMPLES = 320
DEVELOPMENT_OFFSETS = (0, 80, 160, 240)
BANDWIDTH_IDS = {1.5: 0, 3.: 1, 6.: 2, 12.: 3}
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class FrameAlignedCodec:
    """Share an existing NativeCodec model; add no weights or precision changes."""

    def __init__(self, codec, offset_samples):
        if type(offset_samples) is not int or not 0 <= offset_samples < FRAME_SAMPLES:
            raise ValueError('Choose an integer native offset from 0 through 319')
        self.codec = codec
        self._offset_samples = offset_samples
        self.metadata = dict(codec.metadata,
            frame_offset_native_samples=offset_samples,
            frame_offset_native_rate=NATIVE_RATE,
            frame_alignment_code_sha256=IMPLEMENTATION_SHA256,
            frame_alignment_policy='Circular offset before coding; inverse offset before source-rate restoration',
            frame_boundary_policy='Existing reflection to the 320-sample grid after offset',
            frame_alignment_evidence='Experimental framing hypothesis; removal and speech quality unverified')

    @property
    def offset_samples(self):
        return self._offset_samples

    def reconstruct(self, audio, rate, bandwidth_kbps=12.):
        if bandwidth_kbps not in BANDWIDTH_IDS:
            raise ValueError('Use a supported pinned codec bandwidth')
        validate_audio(audio, rate)
        if self.offset_samples == 0:
            return self.codec.reconstruct(audio, rate, bandwidth_kbps=bandwidth_kbps)
        import torch
        channels = audio[:, None] if audio.ndim == 1 else audio
        divisor, rebuilt = gcd(rate, NATIVE_RATE), []
        for channel in channels.T:
            native = resample_poly(channel, NATIVE_RATE//divisor, rate//divisor)
            shifted = np.roll(native, self.offset_samples)
            padded = np.pad(shifted, (0, (-len(shifted)) % FRAME_SAMPLES), mode='reflect')
            with torch.inference_mode():
                result = self.codec.model(torch.from_numpy(padded)[None],
                    bandwidth_id=torch.tensor([BANDWIDTH_IDS[bandwidth_kbps]])).flatten().numpy()
            if len(result) != len(padded) or not np.isfinite(result).all():
                raise RuntimeError('Codec returned invalid samples or the wrong length')
            restored_native = np.roll(result[:len(native)], -self.offset_samples)
            restored = resample_poly(restored_native, rate//divisor, NATIVE_RATE//divisor)[:len(audio)]
            if len(restored) != len(audio) or not np.isfinite(restored).all():
                raise RuntimeError('Codec did not preserve finite samples and sample count')
            rebuilt.append(restored)
        result = np.stack(rebuilt, axis=1)
        result = result[:, 0] if audio.ndim == 1 else result
        gain = min(1., .999/max(1e-12, float(np.max(np.abs(result)))))
        return result*gain, gain
