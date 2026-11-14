"""Bounded local native-donor repair; every proposal requires both final guards."""
from difflib import SequenceMatcher
import numpy as np
from codec_audio import validate_audio
from speech_repair import AlignmentUnavailable, RepairTooLarge, blend_regions
from codec_text import text_units as units


def annotated_units(result, normalize, language, duration):
    values = []
    for chunk in result['chunks']:
        span = chunk['timestamp']
        if (len(span) != 2 or any(value is None for value in span) or not np.isfinite(span).all() or
                not 0 <= span[0] <= span[1] <= duration+.1):
            raise AlignmentUnavailable('A recognized unit has no valid complete-file timestamp')
        values.extend((unit, span) for unit in units(chunk['text'], normalize, language))
    if [v[0] for v in values] != units(result['text'], normalize, language):
        raise AlignmentUnavailable('Offsets do not cover the normalized complete-file sequence')
    return values


def safe_donor_regions(original, native, band, duration, normalize, language,
                       padding=.12, maximum_timing_drift=.25):
    reference = annotated_units(original, normalize, language, duration)
    donor = annotated_units(native, normalize, language, duration)
    candidate = annotated_units(band, normalize, language, duration)
    if not reference or not donor:
        raise AlignmentUnavailable('Original and native speech must be recognized')
    text = lambda values: [v[0] for v in values]
    mapping = {}
    for tag, a, b, c, d in SequenceMatcher(None, text(reference), text(donor), autojunk=False).get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(a, b), range(c, d)):
                first, second = reference[i][1], donor[j][1]
                if abs(sum(first)/2-sum(second)/2) <= maximum_timing_drift:
                    mapping[i] = j
    mapped = set(mapping.values())
    unsafe = [span for index, (_, span) in enumerate(donor) if index not in mapped]
    regions, skipped = [], 0
    for tag, a, b, c, d in SequenceMatcher(None, text(reference), text(candidate), autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if a < b:
            indices = [mapping.get(i) for i in range(a, b)]
            safe = (all(i is not None for i in indices) and
                    indices == list(range(indices[0], indices[0]+len(indices))))
        else:
            anchors = [i for i in (a-1, a) if 0 <= i < len(reference)]
            indices = [mapping.get(i) for i in anchors]
            safe = bool(indices) and all(i is not None for i in indices)
            if safe and len(indices) == 2:
                safe = indices[1] == indices[0]+1
            if safe and a == 0:
                safe = indices[0] == 0
            if safe and a == len(reference):
                safe = indices[-1] == len(donor)-1
            # The inserted candidate's own span is copied, not both anchor words.
            indices = []
        if not safe:
            skipped += 1
            continue
        spans = [span for _, span in reference[a:b]+candidate[c:d]]+[donor[i][1] for i in indices]
        if not spans:
            skipped += 1
            continue
        start, end = max(0., min(s[0] for s in spans)-padding), min(duration, max(s[1] for s in spans)+padding)
        if any(start < right+.04 and end > left-.04 for left, right in unsafe):
            skipped += 1
            continue
        regions.append((start, end))
    if not regions:
        raise AlignmentUnavailable('No safe Native donor region matches the original MMS units')
    return regions, {'unsafe_regions_skipped': skipped, 'maximum_timing_drift_seconds': maximum_timing_drift,
                     'native_original_unit_parity': text(reference) == text(donor),
                     'donor_matching_regions_only': True}


def repair_from_native(original, native, band, native_audio, band_audio, rate, normalize, language,
                       padding=.12, maximum_coverage=.20, maximum_regions=8):
    """Make one bounded proposal; callers must reassess its exact PCM24 audio."""
    validate_audio(native_audio, rate)
    validate_audio(band_audio, rate)
    if native_audio.shape != band_audio.shape:
        raise ValueError('Native and band reconstructions must have the same complete sample grid')
    reference_units = units(original['text'], normalize, language)
    native_units = units(native['text'], normalize, language)
    if not reference_units or not native_units:
        raise AlignmentUnavailable('Native donor and original MMS sequences must be recognized')
    if units(band['text'], normalize, language) == reference_units:
        raise AlignmentUnavailable('No MMS mismatch needs a localized repair')
    duration = len(native_audio)/rate
    for result in (original, native, band):
        if abs(result['duration']-duration) > 1/rate:
            raise ValueError('Alignment and complete waveform durations differ')
    regions, donor_assessment = safe_donor_regions(original, native, band, duration, normalize, language,
                                                  padding=padding)
    if len(regions) > maximum_regions:
        raise RepairTooLarge('Too many distinct intervals need repair')
    output, coverage, gain = blend_regions(band_audio, native_audio, rate, regions,
                                          maximum_coverage=maximum_coverage)
    return np.asarray(output, dtype=np.float32), {'regions_seconds': regions, 'repaired_fraction': coverage,
        'gain': gain, 'padding_seconds': padding, 'maximum_coverage': maximum_coverage,
        'maximum_regions': maximum_regions, **donor_assessment,
        'quality_verified': False, 'content_guard_passed': False, 'api_checked': False,
        'meaning': 'One experimental proposal; requires assessment of the final PCM24 file with both guards.'}
