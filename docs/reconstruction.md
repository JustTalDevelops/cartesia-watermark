# Speech reconstruction

The main upload workflow uses a local Resemble Enhance model with fixed presets. Reconstruction has cleared some tested recordings and failed others. It changes more of the signal than V3; compare the voice and words by listening.

## Install the separate runtime

The editor/web environment uses Python 3.13 and Torch 2.15. Reconstruction uses Python 3.12 and CPU Torch/Torchaudio 2.5.1. Keep them separate.

From the repository root:

```bash
uv venv .cache/resemble-venv --python 3.12
uv pip install --python .cache/resemble-venv/bin/python --torch-backend cpu -r requirements/reconstruction.txt
git clone https://github.com/resemble-ai/resemble-enhance.git .cache/resemble-enhance
git -C .cache/resemble-enhance checkout 8e978149bfe8abab3eb77d965d579a111afdb0ff
```

The wrapper imports directly from this pinned checkout. It does not need the upstream demo application installed.

Download the following files from the [Resemble Enhance model repository](https://huggingface.co/ResembleAI/resemble-enhance/tree/main/enhancer_stage2), preserving their relative paths beneath `.cache/resemble-model/enhancer_stage2/`:

```text
hparams.yaml
ds/G/latest
ds/G/default/mp_rank_00_model_states.pt
```

The checkpoint is approximately 713 MB. The wrapper verifies SHA-256 `f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6` before loading. A different upstream checkpoint fails this check. Assets are downloaded during setup, not during audio processing.

Validate the installation:

```bash
.cache/resemble-venv/bin/python tools/reconstruct_audio.py --check
```

The dependency file records versions from the working local runtime. It is separate from the main project dependencies. Upstream source and model weights retain their own licenses.

## Process audio

```bash
.cache/resemble-venv/bin/python tools/reconstruct_audio.py input.wav output.wav
.cache/resemble-venv/bin/python tools/reconstruct_audio.py input.wav output.wav --preset strong
```

Use a new `.wav` output path; the CLI refuses overwrites. The CLI reads formats supported by libsndfile; the web app also uses FFmpeg to decode uploads. A shared peak gain prevents clipping while preserving stereo balance. Resampling restores the original rate and sample count after model inference.

| Preset | Flow evaluations | Solver | Denoising blend | Prior temperature |
| --- | ---: | --- | ---: | ---: |
| `standard` (default) | 32 | midpoint | 0.1 | 0.5 |
| `full-flow` | 32 | midpoint | 0.1 | 1.0 |
| `strong` | 32 | midpoint | 0.9 | 1.0 |

Processing uses ten-second chunks with one-second overlaps and a fixed seed. The stronger preset had lower speech-recognizer agreement in a recorded trial, so check words as well as naturalness. None of these presets guarantees a negative detector result.

### Check the exact output with Cartesia

The optional verification CLI **uploads candidate recordings to Cartesia**. Install the reconstruction runtime above and configure `CARTESIA_API_KEY` in `.env` or the environment. Its lightweight parent environment needs only NumPy, SoundFile, and Requests. Reuse the app's `.venv` if it exists; for a fresh verification-only setup:

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements/verification.txt
.venv/bin/python tools/verify_reconstruction.py input.wav checked.flac --boundary-safe
```

It tries the fixed `full-flow`, then `strong` presets. A positive result tries the next preset. Only an HTTP 200 response with `detected: false` publishes an output. The published file has exactly the bytes queried: FLAC encoding happens before verification. The file is copied and hashed privately, then published atomically without overwriting an existing output. The command also protects existing audit reports.

This checks the current endpoint response for that file. It does not establish physical watermark erasure, unchanged words, or the result of a different detector or a later query. Listen to the output. The audit records `quality_verified: false` explicitly. The web app remains local and does not run these checks.

Whole recordings must be 5–300 seconds and the final upload at most 32 MB. The command never pads or crops audio to satisfy those limits. FLAC uses 24-bit PCM; WAV retains float samples. API errors, rate limits, malformed responses, and changed files produce no output. Exit status is `0` for a checked negative, `2` when every preset remained detected, and `3` when processing or verification failed. Each invocation gets a new audit in ignored `.cache/verification/`, so a failed check can be retried without discarding its history. Use `--report` to choose a new path, or `--presets full-flow` to test only that preset.

### Experimental chunk correction

`--boundary-safe` avoids a redundant final chunk when the preceding ten-second chunk already covers the recording's end. It also avoids applying an overlap fade to a standalone chunk. Identity-model probes reproduced ending distortion in the upstream scheduling at lengths such as 9.1, 9.5, and 18.5 seconds; the corrected schedule preserves those signals to floating-point roundoff. The per-chunk neural model, resampler, and multi-chunk alignment remain the pinned upstream implementations.

The option is available on both reconstruction CLIs. It is opt-in; the app and the default standalone path retain the reference implementation. Candidate-specific detector and speech-quality evaluation must accompany any decision to promote it. Attribution and the upstream MIT license are in `tools/reconstruction_inference.py` and `tools/third_party/RESEMBLE_LICENSE`.

## Runtime

The web app retains one reconstruction model in an isolated worker between uploads. Model weights, presets, seed, chunking, and waveform processing are unchanged. The standalone CLI starts a fresh runtime for each invocation.

On this machine, a four-second mono fixture took about 11.1 seconds through a fresh CLI process and 6.1–6.4 seconds with a warm worker. Standard → Strong → Standard switching reproduced the original decoded PCM exactly. The first upload still loads the model; longer recordings spend proportionally more time in neural inference, so the relative speedup will be smaller. The worker retains model memory until the server shuts down.

The [runtime report](../results/reconstruction_runtime.json) records these measurements and checks. CPU thread experiments favored the existing four-thread setting, which remains unchanged. These are runtime measurements, not new detector-efficacy results.

## Smoothed-vocoder candidate

`tools/reconstruct_vocoder.py` is a separate experimental candidate under fresh evaluation. It reconstructs a fixed Gaussian-smoothed log-mel spectrogram through upstream NVIDIA BigVGAN. Development checks cleared two difficult recordings; those spent cases do not establish general removal or unchanged speech. The main app continues to use Resemble Enhance.

This candidate uses the main Python 3.13 environment:

```bash
uv pip install --python .venv/bin/python --torch-backend cpu -r requirements/vocoder.txt
git clone https://github.com/NVIDIA/BigVGAN.git .cache/bigvgan
git -C .cache/bigvgan checkout 7d2b454564a6c7d014227f635b7423881f14bdac
```

Download `config.json` and `bigvgan_generator.pt` from the [pinned upstream model revision](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/tree/95a9d1dcb12906c03edd938d77b9333d6ded7dfb) into `.cache/bigvgan-model/`. The wrapper checks both assets and its upstream inference sources against [pinned hashes](../tools/vocoder_assets.json). It performs no downloads during processing. Upstream code, included components, and model weights retain their respective licenses; these weights were not trained by this project.

```bash
.venv/bin/python tools/reconstruct_vocoder.py --check
.venv/bin/python tools/reconstruct_vocoder.py input.wav output.wav --metadata output.json
```

To check this candidate explicitly, use `tools/verify_reconstruction.py input.wav checked.flac --presets smoothed-vocoder --boundary-safe`. The optional preset order `smoothed-vocoder full-flow strong` checks the vocoder first and falls back to the two flow presets only on positive responses. The command loads each runtime only if needed and records which one supplied the checked file. This experimental order is opt-in while evaluation is in progress.

The fixed recipe uses the model's 44.1 kHz, 128-band features, Gaussian sigma 1 in both frequency and time with reflected boundaries, 256-frame synthesis blocks with 32 frames of context, and four CPU threads. Context is discarded at block boundaries without dropping or duplicating samples. Input rate, channels, and sample count are restored; a shared peak gain prevents clipping. Choose new output and metadata paths.
