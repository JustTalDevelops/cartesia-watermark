# Native/TiCodec middle-band preparation

This prepares two fixed mixtures on every one of the 25 spent marked failures.
It has no detector or independent quality results at preparation time. It does
not change the completed study's 19/44 outcome or the declared framing panel.

The motivation is an observed trade-off: the separate TiCodec 1g4r panel passed
operational speech and voice checks on 5/25 sources, rejected 17 for speech and
three for voice estimates. Its first English output passed processing estimates
and received an initial negative detector response. Complete pilot verification
and reference-quality qualification are separate. That single observation
supplies no reliability or erasure claim.

Each source receives both mixtures in this order:

1. Native 12 kbps outside 1–4 kHz, TiCodec inside 1–4 kHz.
2. Native 12 kbps outside 1.5–3.5 kHz, TiCodec inside that narrower band.

The existing sixth-order zero-phase filter produces each mixture. Both stems are
generated audio on the full original sample grid, quantized to PCM24 before
mixing. No original waveform is copied into the mixture. Each Native stem is
regenerated and must reproduce the framing panel's exact zero-offset FLAC before
either treatment can proceed. A mismatch stops generation and remains recorded.

All 50 planned treatments remain, including speech/voice rejections and unfinished
work. TiCodec stems are included regardless of their earlier processing status;
only each final mixed file's actual recognition and voice estimates determine its
operational eligibility. Both existing recognizers, normalization, one-percent
Hindi/Tamil and three-percent other limits, 0.95 speaker similarity and -0.2 MOS
delta remain unchanged. Missing recognizer views never become passing scores.
Source calibration remains the original complete-file observation.

Reference text and evaluation scores do not generate or repair mixtures. All
rendered FLACs remain private, including rejections. This generation command
makes no TTS or detector requests and adds no fresh independent sources. The
assets are the pinned upstream Native codec and already rendered author TiCodec
stems; these are separate from this project's trained-model release.

Actual neural work runs only after the TiCodec generation process exits, one
neural job at a time. A later endpoint phase needs its own prior declaration,
complete-file repeated observations and positive source controls. Independent
reference scoring and human listening remain separate. No proposed mixture is a
validated remover, and detector negatives do not prove physical erasure.

```bash
.cache/resemble-venv/bin/python experiments/probe_ticodec_band.py \
  --completed-study .cache/removal-reliability/codec_profiles_fresh/curated_complete_report.json \
  --ticodec-generation .cache/removal-reliability/ticodec_development/generation.json \
  --frame-generation .cache/removal-reliability/codec_frame_development/generation.json \
  --directory .cache/removal-reliability/ticodec_band_development
```
