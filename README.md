# Cartesia watermark research

Tools for studying Cartesia's audio watermark, running a local detector, and experimenting with speech reconstruction. Includes a local upload app for listening to and downloading processed recordings.

**Removal is experimental.** V3 remains detectable on real recordings. Cartesia returned negative for all **20 marked generated outputs** in the frozen full-flow candidate's [50-recording study](results/reconstruction_reliability.json), but reconstruction can change words and voice quality. The faster vocoder has a fresh Japanese counterexample. Use exact-file verification for a checked result; these tests do not establish universal removal.

**[Download the trained models](https://github.com/JustTalDevelops/cartesia-watermark/releases/tag/models-2026-11-13)** — detector heads, frozen V1–V3 editors, and separate research checkpoints. See the [model guide and catalog](docs/models.md) for contents, loading instructions, and upstream attribution.

## Process a recording

| Method | What it does | Status |
| --- | --- | --- |
| Reconstruction (default) | Rebuilds speech with a local Resemble Enhance model | Mixed detection results; slower, changes more of the signal |
| Full-flow reconstruction | Rebuilds speech from a random latent prior | 20/20 marked generated outputs checked negative; can change spoken words |
| Experimental reconstruction | Rebuilds smoothed speech features with NVIDIA BigVGAN | Faster in the recorded short benchmark; a fresh Japanese case remained detected |
| Stronger reconstruction | Uses more aggressive denoising before reconstruction | Experimental; listen for changes to the voice and words |
| Small edits (V3) | Applies two small waveform networks with a quality guard | Preserves voice fidelity well; unreliable watermark suppression |

The app processes audio locally and never sends uploads to Cartesia. It preserves sample rate, channel count, and sample count, and downloads FLOAT WAV files. Uploads expire after 24 hours and can be deleted immediately. Each method needs its separately documented model assets; the experimental vocoder setup is optional.

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and `ffmpeg`/`ffprobe` on your PATH. Run commands from the repository root:

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements/inference.txt -r requirements/web.txt
```

Reconstruction uses its own Python 3.12 environment and a separately downloaded model. Complete the [reconstruction setup](docs/reconstruction.md) before starting the app:

```bash
.venv/bin/python tools/serve_remover.py --host 127.0.0.1 --port 7860
```

Open **http://127.0.0.1:7860**. See [web app and deployment](docs/web.md) for configuration and limits.

To process a WAV directly with reconstruction:

```bash
.cache/resemble-venv/bin/python tools/reconstruct_audio.py input.wav output.wav
```

The optional [verification CLI](docs/reconstruction.md#check-the-exact-output-with-cartesia) uploads candidates to Cartesia and, by default, saves an output only after two negative responses for those exact file bytes. Its opt-in vocoder → full-flow → strong order produced checked outputs for all 16 planned sources in a [separate study](results/reconstruction_vocoder.json), which includes two reused long recordings. It requires API access; speech quality still needs listening.

The separate experimental V3 CLI needs only the inference dependencies and the bundled checkpoint:

```bash
.venv/bin/python tools/remove_watermark.py input.wav output.wav
```

Read the [editor notes and evaluation](docs/editor.md) before interpreting V3 results.

## Detect locally

```bash
uv pip install --python .venv/bin/python -r requirements/detector.txt
.venv/bin/python tools/detect_local.py recording.wav other.mp3
```

The first run downloads the XLS-R backbone from Hugging Face (about 1.2 GB). The three trained detector heads and configuration are included. Local detection needs no API key; it works on speech clips from roughly three seconds onward.

On the recorded head-to-head test of 96 clips from held-out voices, the local detector reached **95.8% accuracy**, versus **85.4%** for Cartesia's endpoint. These are historical results on generated Sonic speech, not a guarantee for arbitrary recordings. See [detector usage and reproduction](docs/detector.md) and the [full investigation](docs/findings.md).

## Project layout

```text
docs/          Setup, evaluation, research findings, and web design notes
tools/         Processing CLIs, web server, detector, training, and tests
web/           Upload app assets and an example systemd service
requirements/  Dependencies grouped by task
experiments/   Research probes and earlier model approaches
results/       Curated reports, configs, and small release checkpoints
audio/         Reproducible corpus plans; generated audio stays local
```

Generated recordings, feature arrays, caches, environments, logs, and export archives are ignored. The repository includes the small checkpoints needed for local detection and V3 editing; reconstruction weights are downloaded separately.

For tests, training, and the source map, see [development](docs/development.md). API research requires a Cartesia key in `.env`; copy [`.env.example`](.env.example). Inference needs no Cartesia key.
