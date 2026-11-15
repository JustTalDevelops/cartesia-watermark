# TiCodec latent development evaluation

The fixed latent generation panel contains all 25 spent marked failures, three
new decoder cells per source, and one exact quantized control per source. Every
control must match the corresponding original TiCodec FLAC. Generation retains
rejected treatments and does not query the detector or use reference texts.

`experiments/ticodec_latent_generation_report.py` validates every complete file,
sample grid, original recognition cache, processed cache, voice receipt and
unchanged processing limit. Its public export contains numerical evidence and
hashes, with no reference text or transcript. Processing eligibility alone is
not removal evidence.

`experiments/check_ticodec_latents.py` requires the preceding frame and joint
TiCodec detector panels to finish before declaration. Their complete receipts
are bound into the new declaration, and conflicting historical replies stop
the phase. All 75 new latent treatment rows remain in their original source and
mode order. Identical files share observations; an exact match to the already
qualified pilot reuses that earlier evidence without extra requests.

Every new eligible unique file receives two actual service checks, regardless
of the first reply. The first and last source recordings serve as opening and
closing positive controls, with two replies each. The existing shared direct
pacer keeps at least 80 seconds between starts. There are no automatic retries,
proxy rotation or replacement of failed attempts.

```sh
python experiments/ticodec_latent_generation_report.py \
  --directory /path/to/removal-reliability --output /path/to/generation-report.json

python experiments/check_ticodec_latents.py declare \
  --generation /path/to/removal-reliability/ticodec_latent_development/generation.json \
  --completed-study /path/to/removal-reliability/codec_profiles_fresh/curated_complete_report.json \
  --declaration /path/to/endpoint.declaration.json

python experiments/check_ticodec_latents.py detect \
  --declaration /path/to/endpoint.declaration.json --output /path/to/detector.json
```

Reference evaluation follows the complete detector phase and must include every
newly eligible treatment, including detector-positive outputs. It uses the same
language-specific checkpoints, original calibrations and frozen qualification
arithmetic as the completed study. Missing evidence cannot pass. The original
19/44 result remains unchanged, and no new synthesis is required.

These are spent development recordings with no fresh validation sources. The
continuous decoder inputs do not have the original codec's nominal compressed
bitrate. Negative detector replies and estimated quality cannot establish
physical erasure or universal reliability; human listening is still required.
