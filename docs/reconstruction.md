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

## Runtime

The web app retains one reconstruction model in an isolated worker between uploads. Model weights, presets, seed, chunking, and waveform processing are unchanged. The standalone CLI starts a fresh runtime for each invocation.

On this machine, a four-second mono fixture took about 11.1 seconds through a fresh CLI process and 6.1–6.4 seconds with a warm worker. Standard → Strong → Standard switching reproduced the original decoded PCM exactly. The first upload still loads the model; longer recordings spend proportionally more time in neural inference, so the relative speedup will be smaller. The worker retains model memory until the server shuts down.

The [runtime report](../results/reconstruction_runtime.json) records these measurements and checks. CPU thread experiments favored the existing four-thread setting, which remains unchanged. These are runtime measurements, not new detector-efficacy results.
