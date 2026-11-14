"""Bounded word-consensus development profile; no runtime profile is changed.

Only isolated substitutions already supported by the other recognizer before AND
after processing can adjust a comparison transcript. Raw outcomes must remain
recorded. No reference text, detector response, or audio samples enter this rule.
"""
from experiments.codec_consensus_guard import supported_substitutions


def comparison_reference(before, after, anchor_before, anchor_after):
    supported = supported_substitutions(before, after, anchor_before, anchor_after)
    if not before or len(supported) > 2 or len(supported)/len(before) > .03+1e-12:
        return list(before), []
    result = list(before)
    for item in supported:
        result[item['before_index']] = after[item['after_index']]
    return result, supported
