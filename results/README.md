# Release files and recorded evidence

This directory mixes curated files included in Git with ignored local experiment outputs. `.gitignore` lists exactly which reports and checkpoints are publishable.

| Files | Purpose |
| --- | --- |
| `final_detector_r4.json`, `head_conv_heldout_x4c{0,1,2}.pt` | Current local detector config and trained heads |
| `remover_quality.pt`, `remover_quality_v3.pt` | Identical frozen V3 checkpoint; the first is the CLI default alias |
| `remover_quality_v1.pt`, `remover_quality_v2.pt` | Earlier editor components/baselines |
| Matching `*.manifest.json` | Training data and component provenance |
| V2/V3 `*.freeze.json`, `*.release.json` | Historical freeze and release evaluation records |
| `remover_quality_v3_testbatch3_*.json` | Paired quality estimates and actual API responses |
| `sealed*.json`, `h2h4_*.json`, `val_scores_*.json` | Historical detector evaluation |
| `align.json`, `mechanism.json`, `phase2.json`, `detect_cache.json` | Evidence for the watermark investigation |
| `reconstruction_runtime.json` | Reconstruction worker timings and output-equivalence checks |
| `reconstruction_local_shift.json` | A generated unmarked control exposing local-detector distribution shift after reconstruction |
| `reconstruction_reliability.json` | Whole-file endpoint checks, repeats, provenance controls, and speech-quality estimates for the frozen full-flow candidate |
| `reconstruction_vocoder.json` | Frozen vocoder measurements, its Japanese counterexample, and the fixed detector-checked fallback workflow with selected-output quality estimates |
| `reconstruction_vocoder_runtime.json` | Controlled short-fixture timings for full-flow and smoothed-vocoder workers |
| `reconstruction_verification.json` | Real lightweight CLI integration: two negative responses for the same whole-file output, checked again after publication |
| `reconstruction_filtered_prior.json` | Frozen filtered-prior measurements, every-output repeats, checked fallbacks, speech-quality failures, and aggregate private regressions |
| `speech_repair_development.json` | Two spent repair examples, exact-output integration, regional synthesis timings, and the unresolved independent Arabic content regression |
| `speech_repair_runtime.json` | Recognition/timestamp parity, reduced encoder-attention memory, long-source completion and exact packaged audio parity |
| `reconstruction_codec_development.json` | Spent seed/codec/repair probes, Tamil codec negatives, the quality-qualified Arabic detector failure, and independent original-first calibration controls |
| `codec_runtime.json` | Actual lightweight CLI integration for native codec / wide-band fallback: exact development-file reproduction, dual guards, repeated negatives and whole-source positive controls |
| `codec_asr_runtime.json` | Exploratory precision timings, word differences, failed BF16 preprocessing, exact-token short-source encoder reuse and sixteen-original MMS comparison; no processing runtime change or fresh removal evidence |
| `codec_reliability.freeze.json` | Prospective source/evaluation freeze for sixteen new codec-study requests; a protocol, not efficacy evidence |
| `codec_reliability.json` | All sixteen fresh requests: six fully qualified marked transitions, one selected Hindi quality failure, seven uncleared marked cases, and two nominal controls |
| `codec_profiles_reliability.json` | Completed 48-request language-profile study: 19/44 marked sources qualified, 25 failed; four nominal controls and all repeated checks retained separately |
| `codec_frame_development.json` | All 100 frame treatments on 25 spent failures: 60 processing-eligible files, 124 HTTP attempts and one qualified Chinese output; the other detector-negative Arabic output failed its frozen MMS reference check |
| `codec_xcodec2_repair_development.json` | All 75 X-Codec2/repair treatments on 25 spent failures: six sources qualified; French 10 checked negative but failed its Small reference check; 26 HTTP attempts and all eleven eligible outputs reference-scored |
| `ticodec_generation_development.json` | All 25 spent failures rendered with TiCodec: five processing-eligible outputs, seventeen speech rejections and three voice-estimate rejections; no cohort removal qualification |
| `ticodec_pilot_development.json` | One spent English stereo derivative: seven estimated quality criteria passed, two exact-file negatives and four positive source controls; no fresh validation or change to 19/44 |
| `ticodec_joint_development.json` | All 75 TiCodec and middle-band treatments on 25 spent failures: outputs for eight sources qualified, including the prior pilot; three newly detector-negative treatments failed reference checks; 60 new HTTP attempts |
| `dac_pilot_development.json` | All three fixed DAC source outcomes retained: one eligible Tamil output passed all seven quality estimates but remained detected twice; six HTTP attempts, zero qualified removals |
| `ticodec_body_repair_development.json` | All nine generated-donor repairs on three spent hard cases: seven speech rejections, two outputs passing reference checks but remaining detected twice; eight HTTP attempts, zero qualified removals |
| `ticodec_latents_generation_development.json` | Generation-only record of all 75 latent-code treatments and 25 exact paired controls: nineteen processing-eligible outputs, eight voice/naturalness rejections and forty-eight speech rejections; completed checks are in the separate latent development report |
| `ticodec_latents_development.json` | Completed 75-treatment latent panel: ten treatments across five spent sources qualified, all nineteen eligible files reference-scored, 42 HTTP attempts; no fresh validation or change to 19/44 |
| `timing_map_development.json` | Three spent Hindi/Tamil timing-map outcomes: two speech rejections and one quality-qualified output still detected twice; six HTTP attempts and zero qualified removals |
| `xcodec2_prefix_preflight.json` | Three paired synthetic CPU codec timings with exact codes/waveforms and a separate exact RNG check; no real-speech, app-speed or removal result |
| `xcodec2_generation_development.json` | All 25 spent failures: eight processing-eligible outputs, seventeen speech rejections and 27 exact full/prefix channel controls; detector and independent reference qualification pending |
| `codec_repair_development.json` | One spent French counterexample repaired locally: seven estimated quality gates passed, two whole-file negatives and four positive source controls; no change to the fresh-study count |
| `codec_cycles_development.json` | One spent long-file counterexample after two codec passes: all seven estimated quality gates passed, two whole-file negatives and four positive source controls; no fresh independent source |
| `codec_bitrate_development.json` | One spent Korean counterexample: native 3 kbps failed its speech guard; native 6 kbps and both 3/6 kbps band files remained detected twice, including two band files passing all seven estimated quality gates |
| `vocoder_codec_tradeoffs.json` | Three spent sources: Tamil detector-negative quality failures, separate smoothing/repair rejections, and Arabic Mimi speech/detection failures including the additional MMS rejection; every planned variant retained |
| `vocoder_band_development.json` | One spent Tamil source: fixed band-focused smoothing passed seven estimated quality checks and cleared twice with positive source brackets; the stronger variant failed naturalness and was not queried |
| `codec_followup_development.json` | Four spent sources and nineteen treatments: Russian, Korean, and explicitly adjudicated Arabic negatives, exact real-model packaged reproduction, and every remaining speech, naturalness, and detector failure; raw Arabic rejection retained |
| `codec_profile_replay.json` | Nine saved selected files reassessed with the prospective language profiles: seven marked files and two nominal controls passed the current quality estimates; zero new sources or detector requests, with the original Hindi failure and frozen 6/14 result preserved |
| `codec_ta2_followup.json` | One spent Tamil source: eleven rendered variants and two unrendered rejections; the narrow generated inverse mixture passed seven quality estimates and cleared twice with four positive source controls, with exact packaged reproduction; all previous failures retained |
| `codec_inference_followup.json` | Exact real-model parity after deferring unused codec work, short-circuiting rejected candidates and reusing original recognition plus the generated prior; includes French/long-English preflights and a real operational voice-gate acceptance/rejection; no fresh efficacy evidence |
| `codec_precision_development.json` | Eight spent audio variants: INT8 changes one primary decision; packed FP16 matches all tokens and decisions, with a faster exact-file full-wrapper integration using mocked detector replies; outside the active frozen study |
| `model_catalog.json` | Published checkpoint inventory, loaders, aliases, and source/export checksums |

**V3 is not a reliable remover.** Its original small API test did not generalize to subsequent real recordings. See [editor limitations](../docs/editor.md).

Historical reports reference generated audio and intermediate run files that are not included in Git. These records describe those exact original runs; copying them is not equivalent to reproducing the experiment. Regeneration is described in [detector reproduction](../docs/detector.md).

The reconstruction study's [input plan](../audio/reconstruction_reliability_plan.json) includes public source URLs and authored native-generation requests. Repeated queries and originally negative controls are counted separately from positive-to-negative transitions. Private regression recordings, transcripts, and individual fingerprints are excluded from the public report.

The separate [vocoder input plan](../audio/vocoder_reliability_plan.json) distinguishes freshly generated sources from its reused long-recording supplement. A fallback is another treatment of the same recording, not a new test sample. Quality estimates for a discarded primary output must not stand in for the selected fallback's quality.

The [filtered-prior input plan](../audio/filtered_prior_reliability_plan.json) adds a separate fresh cohort and reuses two long sources. Its report retains all four primary counterexamples, every-output repeats, the fixed two-check fallback workflow, and the two selected-file reference-error threshold failures. Private regression evidence is aggregate only.

Selected larger and historical model weights are available in [release downloads](../docs/models.md). Logs, feature arrays, private recording results, optimizer snapshots, and other run outputs remain local. Add new public evidence deliberately rather than unignoring this entire directory.
