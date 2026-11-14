# Codec development results

The frozen [fresh study](codec-study.md) remains **6 qualified marked sources out
of 14**. The follow-ups below reuse its recordings. They add no fresh sources and
do not establish reliable removal on new recordings.

| Follow-up | Observed result |
| --- | --- |
| [French local repair](../results/codec_repair_development.json) | The exact repaired output passed seven estimated quality gates and returned two negative detector responses. |
| [Long English recording](../results/codec_cycles_development.json) | A second native-codec pass preserved the whole 101.12-second recording, passed seven estimated quality gates, and returned two negative responses. |
| [Korean lower bitrates](../results/codec_bitrate_development.json) | Both 3 and 6 kbps band reconstructions passed seven estimated quality gates but stayed detectable twice. The speech-rejected native 3 kbps output was not queried. |
| [Tamil gentle vocoder](../results/vocoder_codec_tradeoffs.json) | Both recordings passed the prospective quality checks but stayed detectable twice. |
| [Tamil stronger vocoder](../results/vocoder_codec_tradeoffs.json) | Midpoint and stronger smoothing cleared detection on the first recording but increased independent recognizer errors. Stronger smoothing also reduced estimated naturalness. These are failed removals. |
| [Tamil band-focused vocoder](../results/vocoder_band_development.json) | Smoothing focused on 1–4 kHz passed seven estimated quality checks on the first recording and returned two negative responses with four positive source controls. The stronger variant failed naturalness and was not queried. This is one spent source. |
| [Tamil separate smoothing and repair](../results/vocoder_codec_tradeoffs.json) | Separate frequency/time smoothing failed the speech guards. A bounded repair passed the processing guards but failed independent speech and naturalness checks. None was detector-tested. |
| [Arabic Mimi codec](../results/vocoder_codec_tradeoffs.json) | The 2.2 kbps output passed the measured quality checks but stayed detectable twice. The 1.1 kbps output failed speech checks. The 1.65 and 1.925 kbps outputs passed seven estimates but failed an additional MMS check. The rejected files were not detector-tested. |

The prospective language profiles differ from the frozen study. Tamil uses
Whisper Large V3 and MMS guards with a one-percent character limit that preserves
vowel marks; Whisper Small provides an additional evaluation. Arabic uses Turbo
and a pinned FastConformer export with a three-percent word limit; Large V3 and
an additional MMS review provide evaluation. Both middle-bitrate Mimi outputs'
MMS reference error increased by 0.0423, exceeding 0.03 despite the other passing
estimates. Checkpoint roles, calibration, hashes, voice
estimates, every planned rejection, and complete detector controls are retained
in the reports. Whisper checkpoints share ancestry.

These are automated estimates, without human listening approval. A negative
detector response does not establish physical erasure. Quality failures remain
failures even when the detector returns negative twice.

The alternative [Mimi weights](https://huggingface.co/kyutai/mimi) are upstream
CC-BY-4.0 assets, separate from the models trained by this project. The smoothing
and codec probes are experiments; they do not change the app's processing methods
or the frozen V3 editor.
