# Frame panel endpoint and reference evaluation

This procedure extends the [declared frame experiment](codec-frame-alignment.md)
on the 25 marked failures of the completed language-profile study. These are
spent development sources. No detector observations have been collected under
this procedure at preparation time. It adds no new TTS requests or fresh sources.

Generation must finish all 100 declared treatments before an endpoint declaration
can be written. The declaration binds the completed generation and study reports,
source bytes, output bytes, processing statuses and implementation hashes. It keeps
all speech and voice rejections. Only `processing_eligible` files reach detection.
Zero-offset outputs must retain the generation panel's historical parity result.

The first included marked source is checked twice as an opening positive control.
Then every eligible treatment is checked twice in source order and offset order
0, 80, 160, 240. Both positive and negative responses require the repeat. The last
included marked source is checked twice as a closing positive control. Identical
encoded candidate files share their two actual observations; all treatment rows
remain and receive no extra independent-source credit. The request ceiling is
twice the number of unique eligible files plus four control checks.

Checks use the complete existing file without cropping, padding or new encoding.
Direct shared pacing remains at least 80 seconds between request starts. Errors,
malformed responses, inconsistent repeats or failed positive controls stop the
phase without automatic retry or replay. A contradiction with a historical
observation of the exact same bytes also stops the phase. Incomplete observations
remain incomplete; a single negative does not qualify a file.

After the entire endpoint phase completes, reference-error scoring covers every
processing-eligible treatment, including those still detected. It uses the same
frozen language-specific evaluator roles, normalization, source calibration and
qualification arithmetic as the completed study. Original calibration and
processing-recognition caches remain bound to their exact files and configuration.
Reference text and evaluation scores cannot generate, repair or choose treatments.
Operational speaker/MOS results are retained as selection estimates, without
presenting them as independent confirmation. Human listening remains separate.

Each of the 100 treatments remains visible in the final report. A spent source
counts as qualified only if at least one treatment has two consistent whole-file
negative responses and passes every frozen quality criterion. Offset treatments,
repeat requests and duplicate bytes are not extra independent successes. Even a
perfect outcome on these spent failures would require a new frozen candidate and
new authored validation cohort. It cannot change the completed 19/44 outcome or
establish physical erasure or universal reliability.

The endpoint runner is [check_codec_frames.py](../experiments/check_codec_frames.py).
It never loads synthesis or recognition models:

```bash
.cache/resemble-venv/bin/python experiments/check_codec_frames.py declare \
  --generation .cache/removal-reliability/codec_frame_development/generation.json \
  --completed-study .cache/removal-reliability/codec_profiles_fresh/curated_complete_report.json \
  --declaration .cache/removal-reliability/codec_frame_development/endpoint.declaration.json

.cache/resemble-venv/bin/python experiments/check_codec_frames.py detect \
  --declaration .cache/removal-reliability/codec_frame_development/endpoint.declaration.json \
  --output .cache/removal-reliability/codec_frame_development/detector.json
```

The declaration also binds original calibration, processing-recognition and voice
receipts. Existing source transcription caches are copied byte for byte into a
new evaluation cache; the completed study's cache is not written. The scorer
blocks HTTP and uses the original pinned evaluator. Its report retains scores for
all eligible offsets regardless of their detector response:

```bash
.cache/resemble-venv/bin/python experiments/codec_frame_report.py score \
  --declaration .cache/removal-reliability/codec_frame_development/endpoint.declaration.json \
  --detector .cache/removal-reliability/codec_frame_development/detector.json \
  --evaluation .cache/removal-reliability/codec_frame_development/evaluation.json

.cache/resemble-venv/bin/python experiments/codec_frame_report.py export \
  --declaration .cache/removal-reliability/codec_frame_development/endpoint.declaration.json \
  --detector .cache/removal-reliability/codec_frame_development/detector.json \
  --evaluation .cache/removal-reliability/codec_frame_development/evaluation.json \
  --output .cache/removal-reliability/codec_frame_development/curated_report.json
```
