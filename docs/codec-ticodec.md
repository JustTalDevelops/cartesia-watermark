# TiCodec development preparation

This is preparation for another codec reconstruction experiment, with no real
speech, detector results or independent quality measurements at preparation time.
It does not change the completed language-profile study's 19/44 outcome.

Recent [latent-watermark research](https://arxiv.org/abs/2609.25830) reports that
robustness differs across codec families. That result concerns other watermark
systems and supplies no evidence about Cartesia's mechanism or TiCodec efficacy
here. It motivates testing a different representation alongside EnCodec and Mimi.

The [author's TiCodec implementation](https://github.com/y-ren16/TiCodec) separates
time-invariant and frame-level codes. This experiment pins its 1g4r configuration,
code revision and [author checkpoint](https://huggingface.co/y-ren16/TiCodec) in
[an asset registry](../experiments/ticodec_assets.json). These upstream weights
were not trained by this project and are not included in its model release.
The inspected code README states Apache 2.0; the author weight repository has no
explicit checkpoint license or model card. No upstream weights are redistributed.

The adapter retains float32 CPU execution and the author model layers. It loads
the verified checkpoint with `weights_only=True`, uses discrete local/global codes,
and never downloads anything during processing. Each complete channel is resampled
to 24 kHz, reflected to the 320-sample grid, encoded and decoded without chunking.
Only the model reflection is discarded before restoring the original grid. A
shared headroom gain preserves channel balance. Frame codes carry 3,000 bits/sec
plus 80 global bits per complete channel, excluding container overhead.

Synthetic-decoder tests cover complete mono/stereo grids, reflection removal,
shared gain and malformed outputs. They establish no speech or removal evidence.
Before speech generation, a separate real-weight synthetic check must match the
author's CPU `VQVAE.encode` and decode API exactly, using shared strictly loaded
weights. It bypasses the author's unrestricted checkpoint loader and retains
weight normalization. This is implementation parity rather than comparison with
the author's CUDA inference script or a quality evaluation.

The first development generation includes every one of the 25 marked failures
from the completed study, one fixed 1g4r treatment each. Inputs are existing
authored TTS, with no new generation charge. Source recognition reuses the
unchanged original calibration. Candidate recognition retains both language-specific
guards and the existing one-percent Hindi/Tamil and three-percent other limits.
Operational voice estimates retain 0.95 speaker similarity and -0.2 MOS delta.
All rendered FLACs and rejections stay private in the manifest. Reference text and
evaluation scores do not generate or repair audio. This command makes no detector
requests; any endpoint/evaluation phase needs a separate prior declaration.

Run real model checks only after the framing generation process has exited, with
one neural job at a time. This experiment is separate from the frame endpoint
panel and cannot replace or remove any of its declared treatments.

```bash
.cache/resemble-venv/bin/python experiments/validate_ticodec.py \
  --directory .cache/removal-reliability/ticodec_synthetic_parity

.cache/resemble-venv/bin/python experiments/probe_ticodec.py \
  --completed-study .cache/removal-reliability/codec_profiles_fresh/curated_complete_report.json \
  --parity .cache/removal-reliability/ticodec_synthetic_parity/parity.json \
  --directory .cache/removal-reliability/ticodec_development
```

Any resulting candidate remains spent-source development and needs a new frozen
validation cohort. Detector negatives and estimated quality cannot establish
physical erasure or universal reliability. Human listening approval is separate.
