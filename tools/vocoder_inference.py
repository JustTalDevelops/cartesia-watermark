"""Bounded vocoder inference with context retained around each output block."""
import numpy as np


def synthesize_chunks(mel, hop, render, frames=256, context=32):
    if hop < 1 or frames < 1 or context < 0 or mel.shape[-1] < 1:
        raise ValueError('Use nonempty features and valid chunk dimensions')
    pieces = []
    for left in range(0, mel.shape[-1], frames):
        right = min(left+frames, mel.shape[-1])
        start, end = max(0, left-context), min(mel.shape[-1], right+context)
        generated = np.asarray(render(mel[..., start:end])).reshape(-1)
        if len(generated) != (end-start)*hop or not np.isfinite(generated).all():
            raise RuntimeError('Vocoder returned invalid samples or the wrong length')
        pieces.append(generated[(left-start)*hop:(right-start)*hop])
    return np.concatenate(pieces)
