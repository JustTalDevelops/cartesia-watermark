# TiCodec and middle-band endpoint declaration

This procedure is declared before its first new request. It includes every
generation row from the completed full-TiCodec and fixed middle-band panels:
25 full renders and 50 band mixtures of the same 25 spent marked failures.
Generation rejections remain visible and never become missing or passing rows.

The completed first-source TiCodec pilot is retained separately: its exact bytes
already have two negatives, four positive source controls and all frozen quality
estimates. That exact source/output pair receives no extra requests or independent
sample credit in this panel. Reuse is explicitly labelled prior evidence, with
its original receipt digest. All other processing-eligible files are pending.

Two opening checks use the first marked source. Every new eligible treatment is
then checked twice in original source order: full TiCodec, wide band, narrow band.
Both positive and negative replies require the repeat. Two closing checks use the
last marked source, even if that source has no eligible treatment. Identical files
share actual observations and retain their rows. The ceiling is twice the number
of unique pending files plus four controls; the completed panels currently yield
28 new eligible unique files and 60 maximum HTTP attempts.

Checks use exact whole PCM24 FLAC bytes, unchanged original sample grids and direct
shared pacing of at least 80 seconds. The existing frame phase may interleave;
both use the same request lock and spacing record. Additional requests can lengthen
that phase without changing either inventory. Errors, inconsistent replies,
contradictions with complete historical exact-file observations or failed positive
controls stop this phase without automatic retry or replay. A shared source-control
contradiction also stops the concurrent frame phase.

Only the already declared processing recognizers and voice estimates determine
eligibility. Reference errors are evaluated after the new endpoint phase, for
every newly eligible treatment including those still detected, using unchanged
frozen roles and qualification arithmetic. The prior pilot retains its original
independent evaluation. Evaluation results cannot change audio or remove rows.

All source, output, assessment, code and completed-input hashes are bound before
the first request and revalidated at phase completion. This is spent development,
with no new TTS or fresh independent sources. It cannot change the 19/44 outcome,
prove physical erasure, provide human listening approval or establish universal
reliability.

```bash
.cache/resemble-venv/bin/python experiments/check_ticodec_panel.py declare \
  --band-generation .cache/removal-reliability/ticodec_band_development/generation.json \
  --completed-study .cache/removal-reliability/codec_profiles_fresh/curated_complete_report.json \
  --band-code-root ../codec-band-prior-20261114 \
  --declaration .cache/removal-reliability/ticodec_joint/endpoint.declaration.json

.cache/resemble-venv/bin/python experiments/check_ticodec_panel.py detect \
  --declaration .cache/removal-reliability/ticodec_joint/endpoint.declaration.json \
  --output .cache/removal-reliability/ticodec_joint/detector.json
```
