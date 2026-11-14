"""Prospective method order learned on spent development recordings.

Only detected utterance language and complete duration choose an order. Reference
text, voice IDs, recognizer evaluation scores, and detector labels do not enter.
This is a candidate for a new frozen study, not held-out efficacy evidence.
"""
from math import isfinite

from codec_settings import MMS_CODES, PRESETS

PROFILE_VERSION = 'codec-language-order-v1'
DEFAULT_ORDER = ('codec-native', 'codec-band')
LANGUAGE_ORDERS = {
    'fr': ('codec-repair', 'codec-native', 'codec-cycle'),
    'ru': ('codec-band-prior-repair', 'codec-native', 'codec-cycle'),
    'ko': ('codec-band-inverse6', 'codec-native', 'codec-band'),
    'ta': ('codec-band-prior', 'codec-band-inverse6-narrow'),
    'ar': ('codec-native', 'codec-band-inverse6-consensus'),
}
LONG_ENGLISH_SECONDS = 60.
LONG_ENGLISH_ORDER = ('codec-cycle', 'codec-native', 'codec-band')


def method_order(language, seconds):
    if language not in MMS_CODES or not isfinite(seconds) or not 5 <= seconds <= 300:
        raise ValueError('Choose a supported detected language and a complete 5–300 second source')
    if language == 'en' and seconds > LONG_ENGLISH_SECONDS:
        return LONG_ENGLISH_ORDER
    return LANGUAGE_ORDERS.get(language, DEFAULT_ORDER)


if any(preset not in PRESETS for order in list(LANGUAGE_ORDERS.values())+[DEFAULT_ORDER, LONG_ENGLISH_ORDER]
       for preset in order):
    raise RuntimeError('A profile references an unavailable reconstruction method')
