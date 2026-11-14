# Speech reconstruction

The main upload workflow uses a local Resemble Enhance model with fixed presets. The default reference preset has mixed detection results. The separately evaluated full-flow preset cleared every processed file in a frozen study, with measurable speech-content risks. Compare the voice and words by listening.

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

Download the following files from the [Resemble Enhance model repository](https://huggingface.co/ResembleAI/resemble-enhance/tree/4e3510ce4a8391159f665903544c5150bee7b2cb/enhancer_stage2), preserving their relative paths beneath `.cache/resemble-model/enhancer_stage2/`:

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

It tries the fixed `full-flow`, then `strong` presets. By default, the CLI requires **two HTTP 200 responses with `detected: false` for the same exact file** before publishing. Any positive response rejects that candidate and tries the next preset, including a positive response after an earlier negative. API errors stop verification without an output. Use `--confirmations 1` for a single check or `--confirmations 3` for three checks; repeated queries add endpoint pacing time, independently of model inference.

The published file has exactly the bytes queried: FLAC encoding happens before verification. The file is copied and hashed privately, then published atomically without overwriting an existing output. The command also protects existing audit reports and records every confirmation. The reusable Python function retains its single-check default for existing callers; the CLI passes its requested count explicitly.

The [recorded CLI integration](../results/reconstruction_verification.json) ran in a parent environment without Torch on a complete 18.96-second authored fixture. It received two separate negative responses and preserved the exact checked FLOAT WAV bytes after publication. This spent fixture is excluded from held-out counts.

This checks the current endpoint response for that file. Repeated responses are not independent recordings or a proof of physical watermark erasure, unchanged words, or future detector behavior. Listen to the output. The audit records `quality_verified: false` explicitly. The web app remains local and does not run these checks.

Whole recordings must be 5–300 seconds and the final upload at most 32 MB. The command never pads or crops audio to satisfy those limits. FLAC uses 24-bit PCM; WAV retains float samples. API errors, rate limits, malformed responses, and changed files produce no output. Exit status is `0` for a checked negative, `2` when every preset remained detected, and `3` when processing or verification failed. Each invocation gets a new audit in ignored `.cache/verification/`, so a failed check can be retried without discarding its history. Use `--report` to choose a new path, or `--presets full-flow` to test only that preset.

### Experimental chunk correction

`--boundary-safe` avoids a redundant final chunk when the preceding ten-second chunk already covers the recording's end. It also avoids applying an overlap fade to a standalone chunk. Identity-model probes reproduced ending distortion in the upstream scheduling at lengths such as 9.1, 9.5, and 18.5 seconds; the corrected schedule preserves those signals to floating-point roundoff. The per-chunk neural model, resampler, and multi-chunk alignment remain the pinned upstream implementations.

The option is available on both reconstruction CLIs. It is opt-in on the command line. The app's default reference method retains the original implementation; its explicit full-flow and stronger-denoising choices use the corrected schedule. Saved results keep their original files and settings. Attribution and the upstream MIT license are in `tools/reconstruction_inference.py` and `tools/third_party/RESEMBLE_LICENSE`.

## Runtime

The web app retains one reconstruction model in an isolated worker between uploads. Model weights, presets, seed, chunking, and waveform processing are unchanged. The standalone CLI starts a fresh runtime for each invocation.

On this machine, a four-second mono fixture took about 11.1 seconds through a fresh CLI process and 6.1–6.4 seconds with a warm worker. Standard → Strong → Standard switching reproduced the original decoded PCM exactly. The first upload still loads the model; longer recordings spend proportionally more time in neural inference, so the relative speedup will be smaller. The worker retains model memory until the server shuts down.

The [runtime report](../results/reconstruction_runtime.json) records these measurements and checks. CPU thread experiments favored the existing four-thread setting, which remains unchanged. These are runtime measurements, not new detector-efficacy results.

## Full-flow reliability study

The [completed report](../results/reconstruction_reliability.json) records 50 whole-file comparisons using the fixed full-flow preset, seed 2035, corrected chunk scheduling, and PCM24 FLAC encoding before endpoint checks. Its [input plan](../audio/reconstruction_reliability_plan.json) records public sources and authored generation requests. The model recipe was frozen before processed labels were observed.

| Sources | Recordings | Original endpoint response | Processed endpoint response |
| --- | ---: | --- | --- |
| Generated marked Sonic speech | 20 | 20 positive | 20 negative |
| Nominally unmarked Sonic 3.5 controls | 5 | 1 positive, 4 negative | 5 negative |
| Public Cartesia audio, model unspecified | 16 | 2 positive, 14 negative | 16 negative |
| Other-provider controls | 9 | 9 negative | 9 negative |

The long supplement includes five 116–252-second recordings from three voices. One extends another recording's text and shares its voice. Twenty core/development repeat queries and five long-output repeat queries all remained negative; three development recordings are excluded from the 50-source corpus. All ten service-control rechecks remained positive. Repeats, related recordings, and originally negative controls do not establish an independent population success rate. A nominally unmarked control's positive response is not proof of a watermark or its subsequent erasure.

Speech quality remains a limitation. The minimum sampled speaker-embedding cosine was 0.961, and the median overall DNSMOS change was −0.052. These descriptive estimates do not guarantee unchanged speech. Whisper Turbo's Arabic reference word error rose from 0% to 13.16%; Tamil also had a substantial increase. A second recognizer and per-recording metrics are retained in the report. No human listening approval is claimed for this candidate, and it does not replace the app's reference default.

The verification CLI checks each new recording's actual final bytes. A finite study cannot prove physical watermark erasure, unchanged words, every future recording, or later detector behavior.

## Filtered-prior candidate

`tools/reconstruct_filtered_prior.py` keeps the original full-band speech condition and low-passes only the waveform used to encode the starting prior. The fixed recipe uses an eighth-order, zero-phase Butterworth filter at 2.3 kHz, prior temperature 0.8, denoising blend 0.1, 32 midpoint evaluations, seed 2035, and corrected chunk scheduling. The [completed study](../results/reconstruction_filtered_prior.json) found four fresh counterexamples to this preset alone; a fixed, exact-file-checked fallback workflow produced negative outputs for all planned sources.

Both spent development recordings cleared Cartesia. In the authored Arabic example, Turbo reference error fell from ten words with full-flow to two; the English regression had no paired transcript differences. The minimum sampled speaker cosine across the two recordings was 0.957, and both overall DNSMOS estimates improved. These development results are not held-out evidence or listening approval. The packaged implementation reproduces both complete development outputs exactly at the sample level.

The frozen [input plan](../audio/filtered_prior_reliability_plan.json) covers twelve marked requests from public voice IDs unused in earlier project plans, two nominal Sonic 3.5 controls, and two reused long recordings. The fourteen fresh requests span twelve voice IDs and eleven requested languages; the English request used a Dutch-primary multilingual voice. The long sources are Spanish (142.88 seconds) and English (252.08 seconds), reused from the earlier supplement rather than counted as fresh voices or generations.

| Sources | Recordings | Original endpoint response | Primary output, first and repeat checks |
| --- | ---: | --- | --- |
| Fresh marked Sonic speech | 12 | 12 positive | 8 negative on both; 4 positive on both |
| Nominal Sonic 3.5 controls | 2 | 1 positive, 1 negative | 2 negative on both |
| Reused long marked sources | 2 | 2 positive | 2 negative on both |

French, Korean, Chinese, and Russian remained detected. Every primary output received an exact-byte repeat, and no responses changed. The nominal French control's original positive response is not proof of a watermark or its subsequent erasure.

The `filtered-prior → full-flow → strong` workflow was sealed before the fresh processed labels. It kept twelve primary outputs and used full-flow for the four remaining detections. Each selected file received two negative responses for its exact PCM24 FLAC bytes; all sixteen sources received a checked output. All planned positive service controls passed. Fallbacks and repeated queries are treatments or checks of the same recordings, not additional samples.

Quality estimates describe the selected files, including the replacements. Fourteen of sixteen met every predeclared estimated threshold: Turbo reference-error increase at most three percentage points, minimum sampled speaker cosine at least 0.95, and overall DNSMOS change at least −0.2. Arabic and Tamil failed the reference-error threshold with increases of 6.45 and 3.55 percentage points. Across all selected files, minimum speaker cosine was 0.961 and median overall DNSMOS change was −0.020. All four replacements met these estimated thresholds; the second recognizer's measurements are also retained. No human listening approval is claimed.

Three spent regression recordings, including two private uploads, were checked separately. All three originals were positive and their filtered-prior outputs were negative on both checks. Their private audio, transcripts, and individual fingerprints are excluded from the public report, and these recordings add nothing to the fresh denominator. An additional integration smoke test ran the actual CLI in a parent environment without Torch: a complete spent generated recording was encoded to PCM24 FLAC, checked negative twice, and published with identical bytes, between positive original-file controls.

It reuses the existing Python 3.12 reconstruction installation and checkpoint. It verifies the reference helpers, model configuration, and relevant upstream source hashes before loading. It does not alter the reference reconstruction implementation or download assets during processing.

```bash
.cache/resemble-venv/bin/python tools/reconstruct_filtered_prior.py --check
.cache/resemble-venv/bin/python tools/reconstruct_filtered_prior.py input.wav output.wav
.venv/bin/python tools/verify_reconstruction.py input.wav checked.flac \
  --presets filtered-prior full-flow strong --boundary-safe
```

The last command uploads candidates to Cartesia and retains the CLI's two-confirmation policy. This preset order is opt-in. The app exposes the candidate as **Balanced reconstruction (experimental)** and keeps reference reconstruction as its default; app uploads remain local and unchecked. These measurements support checking each actual output, not a universal-removal or unchanged-speech guarantee.

The separate [speech repair candidate](speech-repair.md) targets recognizer disagreements with localized vocoder reconstruction. Its development checks passed on the spent Arabic and Tamil regressions, but fresh validation is still pending. It requires an additional local recognizer and does not change the app's default method.

## Smoothed-vocoder candidate

`tools/reconstruct_vocoder.py` is a separate experimental candidate. It reconstructs a fixed Gaussian-smoothed log-mel spectrogram through upstream NVIDIA BigVGAN. Development checks cleared four recordings, but a subsequently generated Japanese recording remained detected. The app offers this as an optional method and keeps the Resemble reference preset as its default.

The vocoder preserved substantially more words in the difficult Arabic development recording than full-flow reconstruction. Its speaker-similarity and DNSMOS estimates declined, including on English recordings. These are model estimates; human listening approval has not been established. A detector-negative response alone does not make this a quality-preserving replacement.

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

To check this candidate explicitly, use `tools/verify_reconstruction.py input.wav checked.flac --presets smoothed-vocoder --boundary-safe`. The optional preset order `smoothed-vocoder full-flow strong` checks the vocoder first and falls back to the two flow presets on positive responses. Install both model runtimes for that order. The command loads each runtime only if needed and records which one supplied the checked file.

The fixed recipe uses the model's 44.1 kHz, 128-band features, Gaussian sigma 1 in both frequency and time with reflected boundaries, 256-frame synthesis blocks with 32 frames of context, and four CPU threads. Context is discarded at block boundaries without dropping or duplicating samples. Input rate, channels, and sample count are restored; a shared peak gain prevents clipping. Choose new output and metadata paths.

In a controlled four-second mono runtime probe under a four-core CPU quota, the median of two warm runs was 4.77 seconds for this candidate and 5.99 seconds for boundary-safe full-flow reconstruction. Cold workers took 6.92 and 10.41 seconds respectively. Cold and warm decoded samples matched exactly within each method. The [runtime metadata](../results/reconstruction_vocoder_runtime.json) records the setup; this short timing probe measured no detector efficacy.

### Fresh evaluation and checked fallback

The [completed vocoder report](../results/reconstruction_vocoder.json) contains 14 newly generated recordings: 12 marked requests from previously unused project voices across 11 languages, plus two matched Sonic 3.5 controls. Two long marked recordings were reused from the full-flow supplement and counted separately. The primary vocoder cleared 11/12 fresh marked sources and both long sources; Japanese remained detected. All ten planned exact-output repeats were negative. The nominal French control was originally positive; its subsequent negative response is not proof of watermark removal.

A fixed `smoothed-vocoder → full-flow → strong` rule was sealed before the fresh processed labels. It produced checked-negative files for all 16 sources, using full-flow only for Japanese. These are the same recordings, with no increase in the sample count for fallback attempts. Positive service controls passed before and after fallback. The study used single checks plus its predeclared repeat schedule; the current CLI's default two-confirmation policy is a subsequent integration change.

Quality estimates describe the actual selected files. The Japanese fallback had unchanged Turbo character error, minimum speaker cosine 0.989, and overall DNSMOS change −0.043. Across all 16 selected files, median paired Turbo disagreement was zero, with maximum word-reference error increase 1.75 percentage points. Minimum speaker cosine was 0.936 and median overall DNSMOS declined by 0.682. The smaller recognizer disagreed more on some cases. These estimates expose a voice-quality tradeoff; no human listening approval or universal success claim is made.
