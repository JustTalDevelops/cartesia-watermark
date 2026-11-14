"""Prospective guard text comparisons; no changes to reconstructed audio."""
from functools import lru_cache
from importlib.metadata import version
import unicodedata

from korean_spacing import spacing_normalized_texts
from speech_repair import disagreement


@lru_cache(maxsize=1)
def chinese_converter():
    if version('OpenCC') != '1.1.9':
        raise RuntimeError('Install OpenCC==1.1.9 for the Chinese writing-system comparison')
    from opencc import OpenCC
    return OpenCC('t2s.json').convert


def text_units(text, normalize, language, converter=None):
    value = normalize(unicodedata.normalize('NFKC', text))
    if language == 'zh':
        value = (converter or chinese_converter())(value)
    return list(''.join(value.split())) if language in ('zh', 'ja') else value.split()


def paired_error(before, after, language):
    if language == 'ko' and before and after:
        reference, hypothesis = spacing_normalized_texts(' '.join(before), ' '.join(after))
        return disagreement(reference.split(), hypothesis.split())
    return disagreement(before, after)
