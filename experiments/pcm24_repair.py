"""Bounded generated-donor blending with exact untouched PCM24 samples.

The upper quantization bin of the existing 0.999 headroom target is accepted.
This prevents a sub-LSB excess from rescaling an otherwise canonical recording.
Coverage, fades, complete-file guards and independent quality checks still apply.
"""
from math import ceil, isfinite

import numpy as np

PCM24_SCALE = 2**23
HEADROOM = ceil(.999*PCM24_SCALE)/PCM24_SCALE


def blend_pcm24_regions(base, donor, rate, regions, fade_seconds=.04, maximum_coverage=.2):
    base, donor = np.asarray(base, dtype=np.float32), np.asarray(donor, dtype=np.float32)
    if (base.shape != donor.shape or base.ndim not in (1, 2) or not len(base) or
            (base.ndim == 2 and base.shape[1] not in (1, 2)) or
            not isinstance(rate, (int, np.integer)) or rate <= 0 or
            not isfinite(fade_seconds) or fade_seconds < 0 or
            not isfinite(maximum_coverage) or not 0 < maximum_coverage <= .2):
        raise ValueError('Use matching mono/stereo PCM24 waveforms and bounded repair settings')
    for samples in (base, donor):
        if (not np.isfinite(samples).all() or np.max(np.abs(samples)) >= 1 or
                not np.array_equal(samples*PCM24_SCALE, np.rint(samples*PCM24_SCALE))):
            raise ValueError('Both inputs must already be finite unclipped PCM24 samples')
    if not regions or len(regions) > 8:
        raise ValueError('Use one to eight bounded repair intervals')
    mask = np.zeros(len(base), dtype=np.float32)
    duration = len(base)/rate
    for start, end in regions:
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration+1e-9:
            raise ValueError('Repair interval is outside the complete recording')
        left, right = int(start*rate), min(len(mask), int(np.ceil(end*rate)))
        local = np.ones(right-left, dtype=np.float32)
        fade = min(int(fade_seconds*rate), len(local)//2)
        if fade:
            ramp = (.5-.5*np.cos(np.linspace(0, np.pi, fade))).astype(np.float32)
            local[:fade], local[-fade:] = ramp, ramp[::-1]
        mask[left:right] = np.maximum(mask[left:right], local)
    coverage = float(np.count_nonzero(mask)/len(mask))
    if not 0 < coverage <= maximum_coverage:
        raise ValueError('Repair would replace too much of the recording')
    weight = mask[:, None] if base.ndim == 2 else mask
    # Identical inputs and zero-weight samples remain exactly identical.
    result = base+(donor-base)*weight
    gain = min(1., HEADROOM/max(1e-12, float(np.max(np.abs(result)))))
    result = result*gain
    if gain == 1. and not np.array_equal(result[mask == 0], base[mask == 0]):
        raise RuntimeError('Untouched PCM24 samples changed during repair')
    return result, {'repaired_fraction': coverage, 'gain': gain, 'peak_headroom_ceiling': HEADROOM,
                    'maximum_coverage': maximum_coverage, 'maximum_regions': 8,
                    'fade_seconds': fade_seconds, 'regions_seconds': regions,
                    'untouched_samples_preserved': gain == 1.,
                    'quality_verified': False, 'content_guard_passed': False, 'api_checked': False}
