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
| `freeze_codec_profiles.py`, `run_codec_profiles.py`, `codec_profile_cache.py`, `codec_profile_evaluation.py` | Seal and run the separate language-profile study, retaining the original study unchanged; fixed method order, bound source caches, operational voice gates and withheld evaluation-ASR reference scoring |
| `codec_study_report.py` | Export every planned source and all seven frozen criteria, rejecting mismatched bytes or receipts and retaining quality/calibration failures after negative detector responses |
| `benchmark_codec_asr.py`, `benchmark_mms_bf16_originals.py`, `codec_asr_runtime_report.py` | Measure CPU precision alternatives and preserve transcript differences and the failed preprocessing attempt; alternatives remain outside the frozen runtime |
| `whisper_encoder_reuse.py`, `benchmark_whisper_encoder_reuse.py` | Prototype FP32 reuse of short-source language/transcription encoder output and compare exact generated tokens, language results and encoder-call counts |
| `codec_ctc_repair.py`, `probe_codec_ctc_repair.py` | Develop one bounded repair of a spent rejected band file using locally matching native-donor units; reject donor errors, timing drift and excessive coverage, reproduce exact failed-file hashes and reassess without changing the frozen study |
| `korean_spacing_diagnostic.py` | Inspect Korean spacing effects using the KsponSpeech character/boundary alignment; preserves every lexical character and never changes frozen study metrics |
| `probe_codec_cycles.py` | Reproduce a spent positive native file exactly, then generate unassessed additional codec passes without detector requests or changes to historical results |
| `codec_cycles_report.py` | Bind second-pass long-file observations to exact bytes and historical source code; voice qualification stays pending until its matching report is complete |
| `probe_codec_bitrate.py`, `probe_codec_bitrate_band.py` | Assess fixed lower codec bitrates and matching wide-band reconstructions on spent sources; no reference text, API requests or fresh efficacy samples |
| `codec_consensus_guard.py` | Diagnose isolated substitutions supported by unchanged context in both other-recognizer transcripts; never changes a processing decision or threshold |
| `codec_word_consensus.py` | Build the explicit bounded Arabic comparison profile while retaining raw disagreements; context support never substitutes for independent quality evaluation |
| `codec_primary_repair.py` | Locate primary-recognizer disagreements in unique independently timed context for bounded generated-donor repair; does not waive transcript differences or use reference/evaluation text |
| `pcm24_repair.py` | Preserve untouched canonical samples and an identical-donor control exactly during bounded blending; accommodates the existing headroom target's upper PCM24 bin without changing speech or coverage limits |
| `probe_vocoder_smoothing.py`, `probe_band_group_delay.py` | Controlled log-mel smoothing and band-delay probes, with whole-file PCM24 outputs and explicit zero-delay parity checks |
| `vocoder_codec_tradeoff_report.py`, `vocoder_band_report.py`, `codec_followup_report.py` | Curate spent codec/vocoder treatments and exact-file integration, preserving quality failures and repeated detector controls |
| `codec_profile_replay_report.py` | Reassess saved selected files with the prospective language profiles, without new synthesis or detector requests; retains both nominal controls and the frozen primary outcome |
| `codec_ta2_followup_report.py`, `codec_inference_followup_report.py` | Retain the failed second-Tamil probes and verify exact-file inference optimizations, including skipped-primary null scores and individual CPU timing limits |
| `codec_precision_report.py` | Recheck every spent FP32/INT8/packed-FP16 transcription comparison, timing and guard decision against private observations and exact input files; curate full-wrapper parity with explicitly mocked detector replies |
| `codec_frame_alignment.py`, `probe_codec_frames.py` | Prepare and generate a fixed frame-offset panel sharing the pinned codec model, only after revalidating the completed study; offsets are undone before rate restoration, and real-model efficacy remains untested; see the [development procedure](../docs/codec-frame-alignment.md) |

The fresh codec study uses [pinned evaluation assets](codec_evaluation_assets.json) and
[text normalization](codec_evaluation_text.py). Chinese reference-error scoring uses
OpenCC 1.1.9 to compare a common writing system, while retaining raw-script errors.
Paired content guards and historical published scores keep their original normalization.
See the [protocol](../docs/codec-study.md) before running these service-calling tools.

Private follow-up experiments using personal uploads remain ignored in `continuous_followup/`.
The current runtime and its setup are documented in [reconstruction](../docs/reconstruction.md).
