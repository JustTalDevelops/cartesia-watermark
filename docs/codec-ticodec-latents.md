# TiCodec local/global quantization experiment

This declares a fixed development ablation before real speech inference. It has
no efficacy or quality results at preparation time. Inputs remain all 25 spent
marked failures of the completed 19/44 study, with no new TTS requests.

The original fully quantized TiCodec panel qualified five outputs for processing;
seventeen failed speech checks and three failed voice estimates. The hypothesis
is that its separate local/global quantizers contribute differently to content,
voice fidelity and any surviving detector signal. The decoder and weights stay
fixed. Four cells isolate those choices:

| Cell | Local decoder input | Global decoder input | Role |
| --- | --- | --- | --- |
| Quantized control | Four RVQ codebooks | Eight global codes | Reproduce the exact original TiCodec file |
| Continuous global | Four RVQ codebooks | Original continuous 128-vector | New treatment |
| Continuous local | Original continuous 512-channel sequence | Eight global codes | New treatment |
| Continuous both | Original continuous sequence | Original continuous vector | New treatment |

Every channel is encoded once and those representations feed all four decoder
cells. The control uses the original code-to-embedding arithmetic and must pass
an actual synthetic comparison with the existing adapter. Every complete speech
control must then match its original TiCodec FLAC SHA before another cell is
assessed. A mismatch stops the whole panel without a waiver.

All 75 new treatments remain, including rejections and incomplete work. The 25
control files are paired implementation controls, not additional independent
sources. All rendered files remain private. Float32, full-channel processing,
reflection to the 320-sample grid, original-rate restoration, natural sample count,
channel count and shared headroom gain retain the existing adapter policy.

The baseline local representation is nominally 3 kbps plus 80 global bits per
complete channel. Continuous local float32 values represent about 1.229 Mbps and
a continuous global vector 4,096 bits per channel. Those cells are not described
as 3 kbps compression. Actual retained audio uses PCM24 FLAC.

Source recognition and both processing guards retain their original file-bound
calibration and comparison rules, including one percent for Hindi/Tamil and three
percent elsewhere. Operational voice limits remain 0.95 speaker similarity and
-0.2 overall MOS delta. Reference text and evaluation scores do not generate,
repair or select these audio treatments. Missing recognizer views never pass.

Inference starts only after the preceding neural job exits. A bounded job may use
the remaining detector-only wait window; it must end before reference evaluation
starts. This generator makes no detector or TTS calls. Any later detector phase
needs its own prior inventory declaration and exact-file controls. Independently
scored reference errors and human listening remain separate.

```bash
.cache/resemble-venv/bin/python experiments/probe_ticodec_latents.py \
  --completed-study .cache/removal-reliability/codec_profiles_fresh/curated_complete_report.json \
  --original-ticodec-generation .cache/removal-reliability/ticodec_development/generation.json \
  --directory .cache/removal-reliability/ticodec_latent_development
```

This experiment cannot revise the original 19/44 result or add fresh efficacy.
Any candidate still requires a new prospective freeze and validation recordings.
Negative replies and estimated quality do not prove physical erasure or universal
reliability. Upstream TiCodec weights remain separate from this project's release.
