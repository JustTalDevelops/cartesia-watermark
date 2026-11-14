"""Prospective ASR scoring normalization; never changes reconstructed audio.

Chinese references and hypotheses use one writing system before CER. Historical
scores keep their original normalization; fresh studies retain raw scores too.
"""
from functools import lru_cache
from importlib.metadata import version
import unicodedata


@lru_cache(maxsize=1)
def chinese_converter():
    if version('OpenCC') != '1.1.9':
        raise RuntimeError('Install the pinned OpenCC==1.1.9 evaluation dependency')
    from opencc import OpenCC
    return OpenCC('t2s.json').convert


def evaluation_units(text, language, normalize, converter=None):
    value = normalize(unicodedata.normalize('NFKC', text))
    if language == 'zh':
        value = (converter or chinese_converter())(value)
    return list(''.join(value.split())) if language in ('zh', 'ja') else value.split()
