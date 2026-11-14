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
| `model_catalog.json` | Published checkpoint inventory, loaders, aliases, and source/export checksums |

**V3 is not a reliable remover.** Its original small API test did not generalize to subsequent real recordings. See [editor limitations](../docs/editor.md).

Historical reports reference generated audio and intermediate run files that are not included in Git. These records describe those exact original runs; copying them is not equivalent to reproducing the experiment. Regeneration is described in [detector reproduction](../docs/detector.md).

The reconstruction study's [input plan](../audio/reconstruction_reliability_plan.json) includes public source URLs and authored native-generation requests. Repeated queries and originally negative controls are counted separately from positive-to-negative transitions. Private regression recordings, transcripts, and individual fingerprints are excluded from the public report.

The separate [vocoder input plan](../audio/vocoder_reliability_plan.json) distinguishes freshly generated sources from its reused long-recording supplement. A fallback is another treatment of the same recording, not a new test sample. Quality estimates for a discarded primary output must not stand in for the selected fallback's quality.

The [filtered-prior input plan](../audio/filtered_prior_reliability_plan.json) adds a separate fresh cohort and reuses two long sources. Its report retains all four primary counterexamples, every-output repeats, the fixed two-check fallback workflow, and the two selected-file reference-error threshold failures. Private regression evidence is aggregate only.

Selected larger and historical model weights are available in [release downloads](../docs/models.md). Logs, feature arrays, private recording results, optimizer snapshots, and other run outputs remain local. Add new public evidence deliberately rather than unignoring this entire directory.
