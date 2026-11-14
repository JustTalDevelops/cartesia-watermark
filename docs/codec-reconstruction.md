# Codec reconstruction candidate

This opt-in CLI reconstructs complete speech with Vocos decoding EnCodec features at 12 kbps. A second preset keeps that reconstruction in the 1000–4000 Hz band and uses a sigma0.5 BigVGAN reconstruction elsewhere. Both presets compare the original and final recording with Whisper Turbo and MMS, rejecting output when either recognizer's disagreement exceeds 3%.

**Fresh validation is pending.** On the two spent Arabic/Tamil development recordings, the packaged verifier reproduced the exact previously measured FLAC bytes and received two negative Cartesia responses per selected file. Arabic's native output failed the MMS guard and was skipped without a detector request; its wider-band fallback passed. Tamil passed natively. Both original-source controls remained positive. See the [runtime evidence](../results/codec_runtime.json) and [earlier development results](../results/reconstruction_codec_development.json).

These are integration checks on two existing recordings. They add no new independent samples. Both recognizers participate in selection, so their agreement is not an independent quality guarantee. The separate reference-error estimates, speaker embeddings and DNSMOS estimates also cannot replace listening or establish physical erasure.

## Install

Reuse the project's Python 3.12 audio environment, or create it for a fresh installation:

```bash
uv venv .cache/resemble-venv --python 3.12
uv pip install --python .cache/resemble-venv/bin/python --torch-backend cpu \
  -r requirements/codec.txt
```

This requirements set does not require Resemble Enhance weights. The wide fallback needs the separate Python 3.13 [BigVGAN installation](reconstruction.md#smoothed-vocoder-candidate). Install its source and weights before using `codec-band`.

Install the pinned Vocos source and codec weights:

```bash
git clone https://github.com/gemelo-ai/vocos.git .cache/vocos
git -C .cache/vocos checkout eb39abfc42c1dee4854d9b10d44dd7d4fd3b0e53
.cache/resemble-venv/bin/hf download charactr/vocos-encodec-24khz \
  --revision 4e61d082c08045a4c11e5b148ad93b1d0c591a14 \
  --local-dir .cache/vocos-encodec-model --include config.yaml pytorch_model.bin
curl --fail --location \
  https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th \
  --output .cache/vocos-encodec-model/encodec_24khz-d7cc33bc.th
```

The runtime loads that local checkpoint rather than allowing EnCodec to download weights while processing. See the upstream [Vocos model card](https://huggingface.co/charactr/vocos-encodec-24khz) and [EnCodec project](https://github.com/facebookresearch/encodec).

Install the recognizers:

```bash
.cache/resemble-venv/bin/hf download openai/whisper-large-v3-turbo \
  --revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 \
  --local-dir .cache/speech-repair-asr \
  --include '*.json' merges.txt model.safetensors
.cache/resemble-venv/bin/hf download facebook/mms-1b-all \
  --revision 3d33597edbdaaba14a8e858e2c8caa76e3cec0cd \
  --local-dir .cache/speech-repair-review-mms \
  --include config.json preprocessor_config.json tokenizer_config.json \
  special_tokens_map.json vocab.json model.safetensors \
  adapter.eng.safetensors adapter.spa.safetensors adapter.fra.safetensors \
  adapter.deu.safetensors adapter.hin.safetensors adapter.jpn.safetensors \
  adapter.ara.safetensors adapter.kor.safetensors \
  adapter.cmn-script_simplified.safetensors adapter.rus.safetensors adapter.tam.safetensors
.cache/resemble-venv/bin/python tools/reconstruct_codec.py --check
.venv/bin/python tools/reconstruct_codec_band.py --check
```

MMS adds roughly 4 GB of weights. Its [pinned model card](https://huggingface.co/facebook/mms-1b-all/blob/3d33597edbdaaba14a8e858e2c8caa76e3cec0cd/README.md) lists **CC-BY-NC-4.0**. These upstream weights are separate from the [models trained by this project](models.md). The [codec manifest](../tools/codec_assets.json), [Whisper manifest](../tools/speech_repair_assets.json), and [BigVGAN manifest](../tools/vocoder_assets.json) pin the files used by this candidate. Processing performs no model downloads.

## Process and verify

For local processing with the native codec:

```bash
.cache/resemble-venv/bin/python tools/reconstruct_codec.py input.wav native.wav \
  --preset codec-native --metadata native.json
```

Use `--preset codec-band` to try the wide fallback directly. A content rejection writes a report, returns exit code 2, and leaves no output recording. Both local presets run without Cartesia requests.

For the fixed native-then-wide order with actual detector checks, configure `CARTESIA_API_KEY` and use the lightweight parent:

```bash
uv pip install --python .venv/bin/python -r requirements/verification.txt
.venv/bin/python tools/verify_reconstruction.py input.wav checked.flac \
  --presets codec-native codec-band --confirmations 2
```

The verifier skips content-rejected files and tries the next preset after a positive detector response. It publishes only the exact file that received the requested negative checks. API errors stop verification without publication. Existing verification defaults remain full-flow followed by strong reconstruction; the web app has no codec option.

The new runtime accepts complete 5–300-second mono/stereo recordings at 8–96 kHz and preserves their sample rate, sample count and channels. The native model runs at 24 kHz, so it cannot retain frequencies above 12 kHz by restoring a higher output rate. The wide fallback uses BigVGAN outside the selected band. At an 8 kHz source rate, its 4 kHz upper edge is Nyquist, so band replacement uses a high-pass filter at the lower edge.

Language is detected from the original's initial Whisper features and forced for each candidate. The installed guard adapters cover English, Spanish, French, German, Hindi, Japanese, Arabic, Korean, Mandarin Chinese, Russian and Tamil. Unsupported languages reject output. Whisper uses native text decoding with timestamps, without word alignment. Both recognizers assess the complete recording, with no artificial word cuts.

Samples are quantized to PCM24 before assessment. A waveform digest binds the guard's result to final encoding. Reports contain counts, scores, model revisions and hashes rather than recognized words, and retain `quality_verified: false`.

The worker reuses the source recognition and native codec file for an immediate content-rejected fallback. It clears these temporary data after ten idle seconds, a different source/job, or shutdown. Direct CLI calls clear them when processing returns. Model weights remain loaded between worker requests; owned worker processes stop together.

## Runtime limits

The recognizers dominate processing time. In this integration run, Tamil's native worker job took about 55 seconds for 28 seconds of speech. Arabic's rejected native job took about 46 seconds, followed by about 61 seconds for its fallback with source data reused. These job timings exclude initial model loading and detector waits, and describe two examples on one CPU machine. They do not establish general throughput or predict fresh-case success.

`reconstruct_codec_band.py` is an internal synthesis worker. Its manifest binds the original and native codec inputs; the owning runtime performs the content assessment before publication. The separate [research mixer](../experiments/codec_band_mix.py) remains available for reproducing earlier band experiments.
