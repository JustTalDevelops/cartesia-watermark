"""Diagnostic Korean spacing alignment; frozen study metrics stay unchanged.

Implements the character/boundary alignment described in KsponSpeech, Appendix
B, Algorithm A2: https://www.mdpi.com/2076-3417/10/19/6936
Only boundaries on matching characters borrow the reference spacing. Inserted
or substituted characters keep their hypothesis boundaries. No lexical
character is added, deleted or changed. Callers normalize Unicode first.
"""
import numpy as np


def spacing_normalized_texts(reference, hypothesis):
    def tokens(text):
        return [(character, offset == 0) for word in text.split() for offset, character in enumerate(word)]

    before, after = tokens(reference), tokens(hypothesis)
    if max(len(before), len(after)) > 6000:
        raise ValueError('Diagnostic alignment exceeds its complete-text limit')
    costs = np.empty((len(before)+1, len(after)+1), dtype=np.int32)
    costs[:, 0] = np.arange(len(before)+1)
    costs[0, :] = np.arange(len(after)+1)
    for i, (first, _) in enumerate(before, 1):
        for j, (second, _) in enumerate(after, 1):
            costs[i, j] = min(int(costs[i-1, j-1])+(first != second),
                              int(costs[i-1, j])+1, int(costs[i, j-1])+1)
    restored = list(after)
    i, j = len(before), len(after)
    while i or j:
        if not i:
            j -= 1
        elif not j:
            i -= 1
        else:
            diagonal = int(costs[i-1, j-1])+(before[i-1][0] != after[j-1][0])
            insertion, deletion = int(costs[i, j-1])+1, int(costs[i-1, j])+1
            if diagonal <= min(insertion, deletion):
                if before[i-1][0] == after[j-1][0]:
                    restored[j-1] = (after[j-1][0], before[i-1][1])
                i, j = i-1, j-1
            elif insertion < deletion:
                j -= 1
            else:
                i -= 1

    def join(values):
        return ''.join((' ' if boundary else '')+character for character, boundary in values).strip()

    normalized_reference, normalized_hypothesis = join(before), join(restored)
    if (''.join(normalized_reference.split()) != ''.join(reference.split()) or
            ''.join(normalized_hypothesis.split()) != ''.join(hypothesis.split())):
        raise RuntimeError('Spacing diagnostics changed lexical characters')
    return normalized_reference, normalized_hypothesis
