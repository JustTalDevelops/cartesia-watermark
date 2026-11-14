"""Locate primary-ASR changes with unique context in independently timed units.

This selects bounded repair intervals, not transcript exceptions. Callers must
use generated donors and reassess the complete repaired PCM24 recording. No
reference text, evaluation recognizer, or detector result enters selection.
"""
from difflib import SequenceMatcher
from math import isfinite


def _checked(values, duration):
    if not values:
        raise ValueError('Timed recognition is empty')
    previous = (0., 0.)
    for _, span in values:
        if (len(span) != 2 or any(not isinstance(x, (int, float)) or not isfinite(x) for x in span)
                or not 0 <= span[0] <= span[1] <= duration + .1):
            raise ValueError('Invalid complete-file timing')
        if span[0] < previous[0] or span[1] < previous[1]:
            raise ValueError('Invalid recognition timing order')
        previous = span
    return [unit for unit, _ in values]


def _unique(sequence, phrase):
    positions = [i for i in range(len(sequence)-len(phrase)+1)
                 if sequence[i:i+len(phrase)] == phrase]
    return positions[0] if len(positions) == 1 else None


def _locate(sequence, left, right):
    a, b = _unique(sequence, left), _unique(sequence, right)
    if a is None or b is None or b < a+len(left):
        return None
    return a+len(left), b


def primary_regions(before, after, original_timed, donor_timed, candidate_timed,
                    duration, context_units=8, padding=.12, maximum_timing_drift=.25,
                    maximum_regions=8):
    if not before or not after or context_units < 2 or duration <= 0 or padding < 0:
        raise ValueError('Nonempty speech and bounded context/timing are required')
    sequences = [_checked(v, duration) for v in (original_timed, donor_timed, candidate_timed)]
    intervals, skipped, changes = [], 0, 0
    for tag, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        changes += 1
        # Both unchanged contexts must be present in the primary transcripts.
        if a < context_units or len(before)-b < context_units or c < context_units or len(after)-d < context_units:
            skipped += 1
            continue
        left, right = before[a-context_units:a], before[b:b+context_units]
        if left != after[c-context_units:c] or right != after[d:d+context_units]:
            skipped += 1
            continue
        locations = [_locate(s, left, right) for s in sequences]
        if any(p is None for p in locations):
            skipped += 1
            continue
        (oa, ob), (na, nb), (pa, pb) = locations
        if (sequences[0][oa:ob] != sequences[1][na:nb] or
                max(ob-oa, nb-na, pb-pa) > max(b-a, d-c)+2*context_units):
            skipped += 1
            continue
        # Include adjacent anchors for an insertion with no original units.
        selected = [v[x:y] if x < y else v[x-1:x+1]
                    for v, (x, y) in zip((original_timed, donor_timed, candidate_timed), locations)]
        spans = [[span for _, span in v] for v in selected]
        if any(not s for s in spans):
            skipped += 1
            continue
        bounds = [(min(s[0] for s in v), max(s[1] for s in v)) for v in spans]
        if any(max(abs(v[0]-bounds[0][0]), abs(v[1]-bounds[0][1])) > maximum_timing_drift
               for v in bounds[1:]):
            skipped += 1
            continue
        # A padded donor interval must also agree in its neighboring CTC words.
        start = max(0., min(v[0] for v in bounds)-padding)
        end = min(duration, max(v[1] for v in bounds)+padding)
        original_context = [u for u, (l, r) in original_timed if l < end and r > start]
        donor_context = [u for u, (l, r) in donor_timed if l < end and r > start]
        if original_context != donor_context:
            skipped += 1
            continue
        intervals.append((start, end))
    if not intervals:
        raise ValueError('No uniquely anchored primary change has a safe generated donor')
    if len(intervals) > maximum_regions:
        raise ValueError('Too many primary change regions')
    return intervals, {'primary_change_regions': changes, 'unsafe_regions_skipped': skipped,
                       'context_units_each_side': context_units,
                       'maximum_timing_drift_seconds': maximum_timing_drift,
                       'padding_seconds': padding, 'maximum_regions': maximum_regions,
                       'donor_matching_regions_only': True, 'reference_text_used': False,
                       'evaluation_recognizer_used': False, 'detector_labels_used': False}
