# experiments/

Historical scripts from the investigation, kept so every number in [the findings](../docs/findings.md) can be
traced to the code that produced it. None of them is needed to run or retrain the current detector; see the
top-level [`README.md`](../README.md) for that. They import shared helpers (`probe`, `cartesia`, ...) from `../tools/`.
Run them from the repository root with `PYTHONPATH=tools`, for example
`PYTHONPATH=tools .venv/bin/python experiments/align.py`.

Scripts that call Cartesia's detector are slow on purpose: the endpoint is rate-limited, so they pace their calls
and keep resumable ledgers in `results/`.

## Mechanism probes (Cartesia's detector as the oracle) — FINDINGS §1–§7

| Script | What it asks |
|---|---|
| `paths.py` | Are SSE and WebSocket output marked too (not just `/tts/bytes`)? |
| `battery.py` | Which everyday edits (gain, filters, codecs, noise, speed, splices) does the mark survive? |
| `phase2.py` | Formats, filter cut-offs, splices and noise levels, at finer steps |
| `mechanism.py` | Phase vs magnitude, sub-band notches, band-only audio, mixing with unmarked speech |
| `align.py` | Is the mark tied to the host's timing? (delay just the 1.5–3.5 kHz band) |
| `look.py` | Raw-sample inspection (quantisation, silence) |

## Signal analysis (no detector calls)

| Script | What it asks |
|---|---|
| `band_shape.py` | Is there a shared in-band spectral shape across marked vs unmarked models? |
| `cepstrum.py` | Echo hiding / fixed delay? (no) |
| `cyclo.py` | A fixed symbol or modulation rate? (no) |
| `known_wm.py` | Is it a published scheme (AudioSeal, WavMark)? (no) |

## Early local detectors (superseded) — FINDINGS §9

| Script | Approach | Why it was dropped |
|---|---|---|
| `surrogate.py` | logistic regression on band spectra | learned model timbre (leave-model-out 36%) |
| `cnn_surrogate.py`, `corpus2.py` | small STFT CNN + its corpus | ~chance on unseen models |
| `train_detector.py` (v2), `train_v3.py`, `train_v4.py`, `train_v5.py`, `eval_all.py` | waveform / STFT CNNs, distillation, rule labels | timbre or "processed audio" shortcuts |
| `wavlm_feats.py`, `wavlm_probe.py`, `wavlm_detector.py` | WavLM layer probes, clip-level logistic regression | first real cross-model signal |
| `wavlm_windows.py`, `wavlm_detector2.py`, `sealed_eval.py` | round 1: windowed WavLM + logistic regression, scored on `audio/sealed` | AUC 0.962 |
| `sealed2_cartesia.py`, `sealed2_final.py` | round 2: Cartesia labels and the one-shot score for `audio/sealed2` | AUC 0.949 |
| `run_r3c.sh` | round 3: WavLM conv heads + contrastive variants, scored on `audio/sealed3` | AUC 0.945 |

`run_r3c.sh` is a record, not something to rerun as-is: `tools/train_head.py` now also trains on corpus7 and
sealed3, which changes its training set.

## Experimental audio editing and reconstruction

These scripts explore candidates, not supported removal methods. Many require generated corpora,
local intermediate checkpoints, optional packages, or API access. Inspect each entry point before running.
Their results do not establish reliable removal on real recordings.

| Scripts | Area |
| --- | --- |
| `bandwidth_probe.py`, `signal_quality_probe.py`, `highband_transform_probe.py`, `upper_gain_probe.py` | Bandwidth, spectral edits, and fidelity |
| `phase_quality_probe.py`, `modulation_quality_probe.py`, `high_phase_cascade_probe.py`, `envelope_probe.py` | Phase and envelope transforms |
| `codec_probe.py`, `denoise_probe.py`, `nmf_band_rebuild.py`, `hybrid_band_probe.py` | Reconstruction alternatives |
| `train_band_rebuilder.py`, `probe_band_rebuilder.py`, `multipass_probe.py`, `refine_probe.py` | Learned band rebuilding and repeated edits |
| `train_encoder_oracle.py`, `train_oracle_head.py`, `train_raw_oracle.py`, `oracle_feedback.py` | Research detector/oracle training |
| `reconstruction_reliability_report.py` | Export paired endpoint and quality measurements from the frozen boundary-correction study; rejects incomplete or mismatched evidence |
| `filtered_prior_report.py` | Export frozen filtered-prior measurements, exact-file fallback checks, and aggregate private regression evidence |
| `speech_repair_development_report.py` | Export two spent repair examples, component timings and exact-output integration; retains the independent Arabic content failure |
| `speech_repair_runtime_report.py` | Validate and export encoder-memory measurements, exact recognition/timestamp comparisons and packaged audio equivalence |
| `codec_development_report.py` | Validate and export spent seed/codec/repair measurements, exact-file endpoint failures and source-calibrated independent recognition checks |
| `codec_band_mix.py` | Reproduce fixed band mixtures from two complete matching reconstructions; writes PCM24 FLAC without quality or detector certification |
| `codec_runtime_report.py` | Verify the implementation commit, exact encoded output, dual content guards and two actual CLI confirmations for each spent integration source |
| `generate_codec_study.py`, `run_codec_study.py`, `score_codec_study.py` | Execute a prospectively frozen fresh codec study with a shared spending ledger, original-first calibration, whole-file repeated detector checks and separately scored final files |
| `codec_study_report.py` | Export every planned source and all seven frozen criteria, rejecting mismatched bytes or receipts and retaining quality/calibration failures after negative detector responses |
| `benchmark_codec_asr.py`, `benchmark_mms_bf16_originals.py`, `codec_asr_runtime_report.py` | Measure CPU precision alternatives and preserve transcript differences and the failed preprocessing attempt; alternatives remain outside the frozen runtime |
| `whisper_encoder_reuse.py`, `benchmark_whisper_encoder_reuse.py` | Prototype FP32 reuse of short-source language/transcription encoder output and compare exact generated tokens, language results and encoder-call counts |
| `codec_ctc_repair.py`, `probe_codec_ctc_repair.py` | Develop one bounded repair of a spent rejected band file using locally matching native-donor units; reject donor errors, timing drift and excessive coverage, reproduce exact failed-file hashes and reassess without changing the frozen study |
| `korean_spacing_diagnostic.py` | Inspect Korean spacing effects using the KsponSpeech character/boundary alignment; preserves every lexical character and never changes frozen study metrics |

The fresh codec study uses [pinned evaluation assets](codec_evaluation_assets.json) and
[text normalization](codec_evaluation_text.py). Chinese reference-error scoring uses
OpenCC 1.1.9 to compare a common writing system, while retaining raw-script errors.
Paired content guards and historical published scores keep their original normalization.
See the [protocol](../docs/codec-study.md) before running these service-calling tools.

Private follow-up experiments using personal uploads remain ignored in `continuous_followup/`.
The current runtime and its setup are documented in [reconstruction](../docs/reconstruction.md).
