"""Explicit experimental word-consensus profile, retaining raw rejections.

Math copied from the bounded 29ab7b3 development profile; no reference text or
watermark labels enter this comparison. The caller restricts it to Arabic.
"""
from difflib import SequenceMatcher

def occurrences(sequence, phrase):
    return [i for i in range(len(sequence)-len(phrase)+1) if sequence[i:i+len(phrase)] == phrase]

def supported_substitutions(before, after, anchor_before, anchor_after, context=2):
    if context < 2:
        raise ValueError('At least two unchanged units are required on each side')
    if not all(isinstance(value, list) and all(isinstance(unit, str) and unit for unit in value)
               for value in (before, after, anchor_before, anchor_after)):
        raise ValueError('Pass normalized nonempty text units as lists')
    supported = []
    for tag, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if (tag != 'replace' or b-a != 1 or d-c != 1 or a < context or c < context or
                b+context > len(before) or d+context > len(after)):
            continue
        prefix, suffix = before[a-context:a], before[b:b+context]
        if prefix != after[c-context:c] or suffix != after[d:d+context]:
            continue
        old_phrase = prefix+before[a:b]+suffix
        new_phrase = prefix+after[c:d]+suffix
        first, second = occurrences(anchor_before, new_phrase), occurrences(anchor_after, new_phrase)
        if (len(first) != 1 or len(second) != 1 or occurrences(anchor_before, old_phrase) or
                occurrences(anchor_after, old_phrase)):
            continue
        supported.append({'before_index': a, 'after_index': c,
                          'anchor_before_index': first[0]+context,
                          'anchor_after_index': second[0]+context, 'context_units_each_side': context})
    return supported

def comparison_reference(before, after, anchor_before, anchor_after):
    supported = supported_substitutions(before, after, anchor_before, anchor_after)
    if not before or len(supported) > 2 or len(supported)/len(before) > .03+1e-12:
        return list(before), []
    result = list(before)
    for item in supported:
        result[item['before_index']] = after[item['after_index']]
    return result, supported
