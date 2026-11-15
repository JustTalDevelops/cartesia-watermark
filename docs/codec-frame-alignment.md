# Codec frame-alignment experiment

This is a preparation for development on spent recordings, separate from the
running [language-profile study](codec-profiles-study.md). No real-model framing
inference, speech-quality measurement or detector result has been collected for
this experiment. Its offline tests use synthetic decoders.

The hypothesis is that changing input alignment relative to a lossy codec's
frame grid may change which signal details survive reconstruction. This may also
damage speech or leave detection unchanged. It is not a reliable removal method.

## Fixed treatment panel

The adapter uses the existing pinned Vocos/EnCodec model at 12 kbps. After source
resampling to 24 kHz, it circularly offsets each complete channel by 0, 80, 160 or
240 native samples. The model uses its existing reflection to the 320-sample
grid. The adapter undoes the offset before restoring the source sample rate,
then retains the complete original sample count and a shared headroom gain.
The offset changes the model's boundary context as well as its frame alignment.

The zero-offset treatment delegates to the existing codec path. All treatments
share one loaded model and retain float32 execution. They add no model weights.
The adapter records its code hash and offset; the public offset property cannot
be reassigned after construction. Its implementation is
[codec_frame_alignment.py](../experiments/codec_frame_alignment.py).

## Development procedure

Actual inference starts only after the current study has finished selection,
withheld transcription scoring, voice scoring and complete-case curation.
Its code, files, qualification rules and outcomes remain unchanged.

The development panel then includes every marked case that failed qualification
in that completed report. Inputs are the existing authored TTS recordings;
personal uploads are excluded. The fixed four offsets above are treatments of
each spent source, not additional independent samples. No new TTS is required.

Before synthesis, record the completed report, input, code, asset and configuration
digests. Generate all four declared treatments for each included source. Retain
speech rejections, quality rejections, baseline mismatches and incomplete work.
For zero-offset comparisons, preserve the exact historical native-file hash
where one was recorded, and verify actual output parity before using the other
treatments as a controlled comparison.

Processing retains both current language-specific recognizers, comparison units
and limits: 1% for Hindi/Tamil and 3% for the other declared profiles. Speaker
similarity must remain at least 0.95 and the DNSMOS overall delta at least -0.2.
Reference text and evaluation-ASR scores do not generate, repair or choose audio.

Any subsequent detector phase must be declared before its first request. It
checks all processing-eligible treatments in the fixed offset order, with two
responses for each exact whole file and opening/closing positive source controls.
Direct shared pacing remains at least 80 seconds. API errors or inconsistent
repeats stop the phase without automatic replay. No crop or pad is added to a
detector input. Voice estimates guide eligibility and are not independent evidence.
Reference-error evaluation follows the declared detector measurements.

Identical encoded files retain their treatment rows and share the same actual
detector observations, without extra requests or independent-sample credit.

## Generate the panel after study completion

From this experiment's checkout, with the existing pinned environments and
assets installed:

```bash
.cache/resemble-venv/bin/python experiments/probe_codec_frames.py \
  --completed-study .cache/removal-reliability/codec_profiles_fresh/curated_complete_report.json \
  --directory .cache/removal-reliability/codec_frame_development
```

The runner checks all completed phases and reproduces the full curated report
before loading models or creating the output directory. It refuses an existing
directory or an output inside the original study directory. Its manifest includes
every failed marked source and all four treatments. Source recognition reuses
the original file-bound calibration; new recognition and voice caches stay in
the development directory. A historical zero-offset mismatch stops the panel.

This command makes no detector or TTS requests. It retains speech and voice
rejections. The retained FLACs are development artifacts; `processing_eligible`
does not mean detector-cleared or independently quality-qualified. A separate
declared phase is still required for endpoint and reference-error measurements.

These development observations cannot rewrite the completed study or establish
fresh efficacy. A candidate formed from them still needs a separately frozen
study on new recordings. Negative detector replies, recognizer agreement and
voice estimates do not prove physical erasure, unchanged speech or universal
reliability. Human listening approval remains separate.
