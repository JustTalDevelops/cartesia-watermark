# Speech repair candidate

This candidate compares recognized content before and after reconstruction. When the difference exceeds 3%, it replaces the mismatched regions with speech synthesized by the smoothed BigVGAN model. The fixed region padding is 80 ms, the cosine crossfade is 40 ms, and repair may cover at most 30% of the recording. The replacement uses reconstructed speech rather than copying the original waveform.

**Fresh validation is pending.** Two spent Arabic/Tamil examples met the primary estimated quality thresholds and received two negative Cartesia responses after repair. The packaged implementation reproduced their complete decoded output exactly, and a real verification CLI integration preserved the exact encoded FLAC bytes. These are development and integration results, not new held-out samples. Whisper-small still showed an Arabic regression; its original error was substantially higher than Turbo's. Neither recognizer agreement nor a negative detector result proves unchanged speech or physical erasure.

A separate architecture review with [Meta MMS](https://huggingface.co/facebook/mms-1b-all) also found an Arabic reference-error increase of 6.45 percentage points. Its original-source errors were about 9.7% for Arabic and 6.5% for Tamil, measured before scoring candidate outputs. The Tamil repaired output increased its MMS error by only 0.59 points. The Arabic failure remains unresolved; matching the guard's own recognizer is insufficient evidence of content fidelity. MMS is currently an evaluation-only dependency and is not part of this runtime.

The [development report](../results/speech_repair_development.json) records the two spent examples, exact-output integration, component timings, and independent-review failure. These examples add no fresh recordings to the removal studies.

## Install

Complete the [Resemble reconstruction](reconstruction.md#install-the-separate-runtime) and [BigVGAN](reconstruction.md#smoothed-vocoder-candidate) installations first. Speech repair adds the pinned recognizer to the existing Python 3.12 environment:

```bash
uv pip install --python .cache/resemble-venv/bin/python -r requirements/speech_repair.txt
.cache/resemble-venv/bin/hf download openai/whisper-large-v3-turbo \
  --revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 \
  --local-dir .cache/speech-repair-asr \
  --include '*.json' merges.txt model.safetensors
.cache/resemble-venv/bin/python tools/reconstruct_speech_repair.py --check
```

The [asset manifest](../tools/speech_repair_assets.json) pins every required file. Processing performs no model downloads. Whisper, Resemble Enhance, and BigVGAN are upstream models, not models trained by this project.

## Process and verify

Local processing accepts complete 5–300-second mono/stereo recordings at 8–96 kHz. It writes a new FLOAT WAV only if the recognizer-consistency guard passes:

```bash
.cache/resemble-venv/bin/python tools/reconstruct_speech_repair.py input.wav repaired.wav \
  --metadata repaired.json
```

For actual detector checks, configure `CARTESIA_API_KEY` and use the [verification parent](reconstruction.md#check-the-exact-output-with-cartesia):

```bash
.venv/bin/python tools/verify_reconstruction.py input.wav checked.flac \
  --presets speech-repair speech-repair-flow speech-repair-strong \
  --confirmations 2 --boundary-safe
```

These profiles use filtered-prior, full-flow, and stronger-denoising bases respectively. Each applies the content guard. A content rejection skips that candidate without a detector request; unavailable verification stops without publishing. Existing CLI defaults remain full-flow followed by strong.

Final samples are quantized to 24-bit PCM before recognition. The verifier checks a digest of the assessed waveform after final encoding, then requires two negative responses for the exact output file. WAV downloads therefore use a float container with samples on the 24-bit grid. Reports contain counts, scores, regions and hashes, without recognized transcripts. Temporary audio is deleted after processing; model workers retain weights between requests and stop together when their owner closes.

The content guard uses Whisper Turbo, which is also one of the evaluation models. It is an estimate rather than independent evidence of unchanged words. Runtime reports retain `quality_verified: false`: there is no automatic human-listening approval or voice-quality certification. Listen to the output.

## Runtime

The extra recognition work makes this path slower than base reconstruction. Before the encoder-memory optimization below, the packaged worker took about 128 and 145 seconds on the two roughly 28-second development recordings. These timings describe those recordings, not general throughput.

The recognizer now discards unused encoder-attention diagnostics after computing each layer's output. Decoder cross attention remains available for word alignment. Complete text, language and word timestamps matched exactly in the 28-second and 60-second comparisons; peak memory fell from 9.3 to 4.9 GiB and from 14.7 to 5.0 GiB. Recognition time on those short fixtures was approximately unchanged.

The optimized recognizer completed a whole 252-second recording in about 92 seconds at 4.9 GiB peak memory in the known-English timing probe. The earlier baseline was stopped after about 324 seconds at 32.1 GiB, without a completed recognition result. These are observations on one machine; the long case has no completed baseline transcript to compare. The [runtime report](../results/speech_repair_runtime.json) separates recognition parity, the incomplete baseline, and packaged-output parity.

The automatic-language preflight also exposed missing language tags in long-form word chunks. The runtime now uses the pinned model's initial-feature language detector when those tags are absent. Conflicting or unknown tags still reject the recording. The same complete long source then returned English with valid word timings in about 103 seconds at 4.9 GiB. The two packaged output comparisons were measured before this separate language fix; its missing-tag/conflict cases and the actual complete long source were checked afterward.

The donor renderer skips unused global 256-frame synthesis blocks while retaining each selected block's original 32-frame context. A controlled probe measured about 40.2 to 19.5 seconds for Arabic and 38.7 to 24.9 seconds for Tamil, with identical final FLAC bytes. These are two observations on one machine, not an overall pipeline speed guarantee. Tests also compare used samples with full rendering across sample rates and channel counts.

`reconstruct_vocoder_regions.py` is an internal donor renderer. Its intermediate waveform has omitted blocks and is not a final recording or detector test input. Only the complete repaired recording is checked.

## Additional development checks

These follow-ups reused the same spent recordings and add no fresh samples:

| Probe | Result |
| --- | --- |
| Union of Turbo/MMS disputed regions, smoothed donor | Arabic MMS reference-error increase fell to 3.23 points, still above the 3-point threshold |
| Same union with an unsmoothed donor | MMS increase fell to 1.61 points; Whisper-small still increased by 6.45 points |
| Union of Small/Turbo disputed regions | Rejected because replacement exceeded the fixed 30% limit; no output or detector query |
| Existing full-flow repair profile | Arabic passed its own guard but MMS increased by 9.68 points; Tamil was rejected at 3.012% guard disagreement |
| Existing stronger repair profile | Arabic passed its own guard but MMS increased by 8.06 points; Tamil was rejected at 3.61% disagreement |
| Complete unsmoothed vocoder | Arabic met the estimated quality criteria, but remained detected on both endpoint checks |
| Complete vocoder with fixed sigma0.5 smoothing | Both Arabic and Tamil met the estimated quality criteria; both remained detected on both checks |

Both complete original recordings also remained detected on two checks each after encoding to PCM24 FLAC without reconstruction. Opening and closing service controls stayed positive. The complete unsmoothed vocoder used the same FLAC encoding; its result therefore preserves a counterexample with matched file formats.

These failures remain in the [development report](../results/speech_repair_development.json). They prevent a claim of reliable general removal, even though some reconstructed outputs check negative.

The separate [codec development report](../results/reconstruction_codec_development.json) retains later probes on these same two spent sources. Vocos with EnCodec at 12 and 6 kbps passed all seven estimated quality criteria on Tamil and checked negative twice at each setting. Both native Arabic outputs failed the unchanged quality criteria. A fixed rule that kept Tamil native and repaired Arabic with a sigma0.5 donor brought both selected files within those criteria, but Arabic remained detected twice. Opening and closing original-source controls remained positive. This rule has not been promoted to the runtime or tested on fresh recordings.

Changing only the original file format or round-tripping its sample rate through 24 kHz left both originals detectable twice. Four full-flow random seeds per source produced no output that passed all seven quality criteria; those eight variants were rejected before detector requests. These are treatments of two recordings, not new independent samples. The codec models operate at 24 kHz and cannot retain frequencies above 12 kHz merely by restoring the original sample rate. All codec and additional recognition models are upstream assets rather than project-trained releases.
