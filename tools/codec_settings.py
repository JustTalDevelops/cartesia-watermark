"""Lightweight settings for opt-in codec reconstruction; safe for the API parent."""
PRESETS = ('codec-native', 'codec-band')
MAX_DISAGREEMENT = .03
MMS_CODES = {'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu', 'hi': 'hin',
             'ja': 'jpn', 'ar': 'ara', 'ko': 'kor', 'zh': 'cmn-script_simplified',
             'ru': 'rus', 'ta': 'tam'}
SETTINGS = {'codec_bandwidth_kbps': 12., 'codec_sample_rate': 24000,
            'codec_bandwidth_limit_hz': 12000, 'fallback_band_hz': [1000, 4000],
            'fallback_filter_order': 6, 'fallback_zero_phase': True,
            'fallback_upper_edge_at_nyquist': 'High-pass at the lower edge',
            'fallback_mel_sigma_frequency': .5, 'fallback_mel_sigma_time': .5,
            'maximum_paired_disagreement_each_recognizer': MAX_DISAGREEMENT,
            'sample_grid_before_guard': 'PCM_24', 'threads': 4}
