"""Render selected global vocoder blocks without changing their context.

The returned waveform is an unscaled intermediate, with zeros in omitted
blocks. Only the requested intervals are valid for use in a speech repair.
"""
from numbers import Integral

import numpy as np


def synthesize_regions(mel, hop, render, regions, frames=256, context=32):
    dimensions = (hop, frames, context)
    if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in dimensions) or
            hop < 1 or frames < 1 or context < 0 or mel.shape[-1] < 1):
        raise ValueError('Use nonempty features and valid integer chunk dimensions')
    samples = mel.shape[-1]*hop
    blocks = set()
    for region in regions:
        if len(region) != 2:
            raise ValueError('Each sample interval needs two bounds')
        left, right = region
        if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in region) or
                not 0 <= left < right <= samples):
            raise ValueError('Sample intervals must lie inside the synthesized waveform')
        blocks.update(range(left//(frames*hop), (right-1)//(frames*hop)+1))
    if not blocks:
        raise ValueError('Choose at least one sample interval')
    output = None
    for block in sorted(blocks):
        left = block*frames
        right = min(left+frames, mel.shape[-1])
        start, end = max(0, left-context), min(mel.shape[-1], right+context)
        generated = np.asarray(render(mel[..., start:end])).reshape(-1)
        if (len(generated) != (end-start)*hop or
                not np.issubdtype(generated.dtype, np.floating) or not np.isfinite(generated).all()):
            raise RuntimeError('Vocoder returned invalid samples or the wrong length')
        if output is None:
            output = np.zeros(samples, dtype=generated.dtype)
        elif generated.dtype != output.dtype:
            raise RuntimeError('Vocoder changed its output dtype between blocks')
        output[left*hop:right*hop] = generated[(left-start)*hop:(right-start)*hop]
    return output, tuple(sorted(blocks))
