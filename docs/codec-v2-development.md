# Codec V2 development branch

This branch contains prospective processing changes. It has no new held-out
removal result. The original codec study remains six fully qualified marked
sources out of fourteen. Detector negatives and automated quality scores do not
establish universal removal.

The current development implementation adds:

- FP32 Whisper short-source encoder reuse. Language detection and transcription
  share an encoder result on inputs up to thirty seconds; longer files keep the
  complete-input path.
- A language-specific second recognizer: FP32 SenseVoice for English, Japanese,
  Korean and Chinese; FP32 MMS for the other seven supported languages.
- Chinese writing-system normalization, Korean spacing-normalized word
  comparisons, and preservation of essential Hindi/Tamil combining vowel signs
  and viramas. The three-percent paired disagreement limit remains in place.
- `codec-repair`: one wide-band reconstruction, optionally repaired from local
  native-codec regions whose recognized units match the original. At most eight
  proposed regions and twenty percent of the recording may be replaced. Timing
  drift and neighboring donor errors can prevent a repair.
- `codec-cycle`: a second native-codec pass, with the same bounded repair option
  when MMS alignment is available.

Both recognizers reassess the final PCM24 file. A failed assessment leaves no
output recording. Reference text and evaluation-only models never enter
processing. The verification CLI checks whole files and publishes only the exact
bytes receiving its requested negative responses.

## Additional setup

First install the existing [codec prerequisites](codec-reconstruction.md), plus
this branch's updated `requirements/codec.txt`. This pins sherpa-onnx 1.13.8 and
OpenCC 1.1.9. The new presets also require the existing BigVGAN installation when
their band reconstruction is used.

Download the FP32 SenseVoice export from the `source_url` in
[`tools/codec_assets.json`](../tools/codec_assets.json). Verify its
`archive_sha256` before extraction, then extract into
`.cache/sensevoice-review-f32/`. The resulting directory must contain the exact
`model.onnx`, `tokens.txt` and `LICENSE` files listed in that manifest.

SenseVoice uses the upstream
[FunASR Model Open Source License Agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
MMS uses CC-BY-NC-4.0. These are upstream models, separate from the checkpoints
trained and published by this project. Processing performs no downloads.

Validate the installed profiles:

```bash
.cache/resemble-venv/bin/python tools/reconstruct_codec.py --check
```

Try a local development preset:

```bash
.cache/resemble-venv/bin/python tools/reconstruct_codec.py input.wav candidate.wav \
  --preset codec-repair --metadata candidate.json
```

A content rejection returns exit code 2. Detector checking remains a separate,
explicit upload through the verification CLI. The web interface has no V2 option.

## Evidence so far

The packaged French repair reproduced the exact development file that passed
all seven estimated quality gates and returned detector-negative twice, with
positive original controls before and after. This is a spent development example.
The Korean native and band files both remained detected despite passing the new
speech guards. Arabic local repair was refused because no safe donor region
matched the original. The second-pass long-file detector checks and further
language probes are still underway.

The implementation passes 169 existing and added tests. Fresh validation must
freeze this implementation, model profiles and evaluation policy before creating
new source recordings. Historical study export validates the recorded Git
snapshot; reproducing its processing requires checking out that historical code.
