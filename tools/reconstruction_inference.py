"""Boundary-safe orchestration for the pinned Resemble Enhance model.

Adapted from resemble_enhance/inference.py at revision
8e978149bfe8abab3eb77d965d579a111afdb0ff (MIT; third_party/RESEMBLE_LICENSE).
The resampler, per-chunk model, and multi-chunk alignment/merge are unchanged.
Only necessary chunks are rendered; a single chunk bypasses overlap fades.
"""
from functools import partial


def chunk_starts(length, chunk_length, overlap_length):
    if length < 1 or chunk_length < 1 or not 0 <= overlap_length < chunk_length:
        raise ValueError('Use nonempty audio and valid chunk/overlap lengths')
    for start in range(0, length, chunk_length-overlap_length):
        yield start
        if start+chunk_length >= length:
            break


def render_chunks(audio, chunk_length, overlap_length, render, merge):
    """Render complete coverage once and avoid fading a standalone recording."""
    chunks = [render(audio[start:start+chunk_length])
              for start in chunk_starts(len(audio), chunk_length, overlap_length)]
    if len(chunks) == 1:
        return chunks[0]
    return merge(chunks, chunk_length, chunk_length-overlap_length, length=len(audio))


def inference(model, audio, sample_rate, device, chunk_seconds=10., overlap_seconds=1.):
    from resemble_enhance.inference import (inference_chunk, merge_chunks,
                                           remove_weight_norm_recursively)
    from torchaudio.functional import resample

    remove_weight_norm_recursively(model)
    rate = model.hp.wav_rate
    audio = resample(audio, orig_freq=sample_rate, new_freq=rate,
                     lowpass_filter_width=64, rolloff=0.9475937167399596,
                     resampling_method='sinc_interp_kaiser', beta=14.769656459379492)
    chunk_length, overlap_length = int(rate*chunk_seconds), int(rate*overlap_seconds)
    output = render_chunks(audio, chunk_length, overlap_length,
                           render=lambda wave: inference_chunk(model, wave, rate, device),
                           merge=partial(merge_chunks, sr=rate))
    return output, rate
