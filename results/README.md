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
| `model_catalog.json` | Published checkpoint inventory, loaders, aliases, and source/export checksums |

**V3 is not a reliable remover.** Its original small API test did not generalize to subsequent real recordings. See [editor limitations](../docs/editor.md).

Historical reports reference generated audio and intermediate run files that are not included in Git. These records describe those exact original runs; copying them is not equivalent to reproducing the experiment. Regeneration is described in [detector reproduction](../docs/detector.md).

Selected larger and historical model weights are available in [release downloads](../docs/models.md). Logs, feature arrays, private recording results, optimizer snapshots, and other run outputs remain local. Add new public evidence deliberately rather than unignoring this entire directory.
