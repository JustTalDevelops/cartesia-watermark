"""Deterministic alignment and blending for experimental speech repair."""
from difflib import SequenceMatcher
import hashlib
import unicodedata

import numpy as np

PRESETS = {'speech-repair': 'filtered-prior', 'speech-repair-flow': 'full-flow',
           'speech-repair-strong': 'strong'}
MAX_DISAGREEMENT = .03
MAX_COVERAGE = .30


class AlignmentUnavailable(ValueError):
    pass


class RepairTooLarge(ValueError):
    pass


def units(text, normalize, language):
    value = normalize(unicodedata.normalize('NFKC', text))
    return list(''.join(value.split())) if language in ('zh', 'ja', 'yue') else value.split()


def disagreement(before, after):
    if not before or not after:
        return None
    previous = list(range(len(after)+1))
    for number, first in enumerate(before):
        current = [number+1]
        for offset, second in enumerate(after):
            current.append(min(previous[offset+1]+1, current[-1]+1, previous[offset]+(first != second)))
        previous = current
    return previous[-1]/len(before)


def aligned_regions(before, after, duration, normalize, language, padding=.08):
    def annotated(result):
        values = []
        for chunk in result['chunks']:
            values.extend((unit, chunk['timestamp']) for unit in units(chunk['text'], normalize, language))
        if [v[0] for v in values] != units(result['text'], normalize, language):
            raise AlignmentUnavailable('Word timestamps do not cover the recognized text')
        return values

    original, edited = annotated(before), annotated(after)
    regions = []
    for tag, a, b, c, d in SequenceMatcher(None, [v[0] for v in original],
                                          [v[0] for v in edited], autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        spans = [v[1] for v in original[a:b]+edited[c:d]]
        if (not spans or any(len(s) != 2 or s[0] is None or s[1] is None or
                             not np.isfinite(s).all() or not 0 <= s[0] <= s[1] <= duration+.1 for s in spans)):
            raise AlignmentUnavailable('A mismatched speech unit has no valid time interval')
        regions.append((max(0., min(s[0] for s in spans)-padding),
                        min(duration, max(s[1] for s in spans)+padding)))
    return regions


def blend_regions(base, donor, rate, regions, fade_seconds=.04, maximum_coverage=MAX_COVERAGE):
    if (base.shape != donor.shape or base.ndim not in (1, 2) or not len(base) or
            not np.isfinite(base).all() or not np.isfinite(donor).all() or rate <= 0):
        raise ValueError('Use finite, matching waveforms and a valid sample rate')
    if not regions:
        raise AlignmentUnavailable('No mismatched regions could be repaired')
    mask = np.zeros(len(base), dtype=np.float32)
    duration = len(base)/rate
    for start, end in regions:
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration+1e-9:
            raise AlignmentUnavailable('Speech interval is outside the recording')
        left, right = int(start*rate), min(len(mask), int(np.ceil(end*rate)))
        local = np.ones(right-left, dtype=np.float32)
        fade = min(int(fade_seconds*rate), len(local)//2)
        if fade:
            ramp = (.5-.5*np.cos(np.linspace(0, np.pi, fade))).astype(np.float32)
            local[:fade], local[-fade:] = ramp, ramp[::-1]
        mask[left:right] = np.maximum(mask[left:right], local)
    coverage = float(np.count_nonzero(mask)/len(mask))
    if not 0 < coverage <= maximum_coverage:
        raise RepairTooLarge('Repair would replace too much of the recording')
    weight = mask[:, None] if base.ndim == 2 else mask
    result = base*(1-weight)+donor*weight
    gain = min(1., .999/max(1e-12, float(np.max(np.abs(result)))))
    return result*gain, coverage, gain


def pcm_digest(audio):
    return hashlib.sha256(np.ascontiguousarray(audio, dtype='<f4').tobytes()).hexdigest()
