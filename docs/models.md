# Trained model downloads

The [trained-model release](https://github.com/JustTalDevelops/cartesia-watermark/releases/tag/models-2026-11-13) publishes this project's selected checkpoints as downloadable assets. The primary detector heads and frozen editor releases are also included directly in Git. Larger and historical research checkpoints live in release assets to keep the repository small.

**V3 is not a reliable watermark remover.** It still flags Cartesia's detector on real recordings. Research editors, historical detectors, and teacher models are included for inspection and reproduction; publishing them does not establish reliable removal or new evaluation results.

## Packages

| Download | Contents |
| --- | --- |
| [detector-r4.tar.gz](https://github.com/JustTalDevelops/cartesia-watermark/releases/download/models-2026-11-13/detector-r4.tar.gz) | The three trained round-4 convolutional heads and calibrated detector configuration |
| [editors-v1-v3.tar.gz](https://github.com/JustTalDevelops/cartesia-watermark/releases/download/models-2026-11-13/editors-v1-v3.tar.gz) | Frozen V1/V2 weights, V3's combined weights/policy, default alias, manifests, and original freeze/release records |
| [research-editors.tar.gz](https://github.com/JustTalDevelops/cartesia-watermark/releases/download/models-2026-11-13/research-editors.tar.gz) | Selected spectral/waveform editors, band rebuilders, adaptive training checkpoints, and labelled derived variants |
| [research-detectors.tar.gz](https://github.com/JustTalDevelops/cartesia-watermark/releases/download/models-2026-11-13/research-detectors.tar.gz) | Earlier feature heads, CNN experiments, WavLM fine-tunes, and round-2/3 detector configurations |
| [research-oracles.tar.gz](https://github.com/JustTalDevelops/cartesia-watermark/releases/download/models-2026-11-13/research-oracles.tar.gz) | Fine-tuned XLS-R and AudioSeal-based classifiers used as training teachers |
| [model-catalog.json](https://github.com/JustTalDevelops/cartesia-watermark/releases/download/models-2026-11-13/model-catalog.json) | Every published checkpoint's filename, loader, configuration, training step/epoch where recorded, source/export hashes, and aliases |
| [SHA256SUMS](https://github.com/JustTalDevelops/cartesia-watermark/releases/download/models-2026-11-13/SHA256SUMS) | SHA-256 checksums for release archives and the catalog |

The same [catalog is tracked in Git](../results/model_catalog.json). Identical model configurations/tensors are deduplicated; the catalog maps other checkpoint names to their published equivalents. V3 is a composition of trained components and a fixed policy, rather than an independently trained third network. Derived weight/policy variants are labelled accordingly.

Optimizer states, scratch/probe snapshots, and selected checkpoints that retained their initialization without a new training update are omitted. Full training corpora, recordings, and private manifests are not included. Where available, original training-manifest hashes remain in the exported checkpoints and catalog.

## Download and check

Download an archive from the release page, or use the GitHub CLI from the repository root:

```bash
mkdir -p artifacts/models
gh release download models-2026-11-13 --repo JustTalDevelops/cartesia-watermark \
  --pattern research-editors.tar.gz --pattern SHA256SUMS --dir artifacts/models
(cd artifacts/models && sha256sum --ignore-missing -c SHA256SUMS)
tar -xzf artifacts/models/research-editors.tar.gz -C artifacts/models
(cd artifacts/models/research-editors && sha256sum -c SHA256SUMS)
```

Each archive contains a named top-level directory, `results/` checkpoint files, `models.json`, a model card, checksums, and third-party notices. Downloaded archives can stay below the ignored `artifacts/` directory.

## Use the weights

The default detector and V3 editor already work with the checkpoints in a normal clone; see [detector usage](detector.md) and [editor usage](editor.md). To restore their files from a release archive, extract it and copy its `results/` contents into the checkout's `results/` directory.

Research audio editors use the same loader and CLI, with an explicit path:

```bash
.venv/bin/python tools/remove_watermark.py input.wav output.wav \
  --checkpoint artifacts/models/research-editors/results/remover_adaptive25balanced.pt
```

That example selects an experimental candidate with different edit-energy limits from V3. Listen to the output; it is not a recommended or validated replacement for the default workflow.

Historical detectors and oracles require the loader identified in their package's `models.json` or the global catalog. Plain head checkpoints are state dictionaries; the head's `feature_set` must match the feature extractor. Legacy CNNs also have different preprocessing and input lengths, documented in their source scripts. Fine-tuned backbone loaders may still fetch upstream initialization assets before loading these weights.

Research inference exports keep the original format, model tensors, and configuration while removing optimizer/RNG state and machine-specific training arguments. Every exported tensor and configuration was compared with the selected source checkpoint; audio-editor network outputs were also compared. The frozen primary files remain byte-identical to their original releases.

## What was trained here

The convolutional detector heads and small audio editors were trained for this investigation. WavLM/XLS-R/AudioSeal research checkpoints fine-tune upstream backbones; the catalog identifies them as such. They are not Cartesia's original detector weights.

The web app's reconstruction model is **upstream Resemble Enhance**, not a model trained by this project. Its weights are not republished here; use the [separate reconstruction setup](reconstruction.md). Other unmodified upstream model downloads are likewise excluded.

The experimental [smoothed-vocoder candidate](reconstruction.md#smoothed-vocoder-candidate) uses upstream NVIDIA BigVGAN. Its source and weights are pinned and downloaded separately; they are not part of this project's trained-model release.

The [third-party notices](model-licenses/THIRD_PARTY_NOTICES.md) document backbone origins, modifications, and included licenses. In particular, the WavLM fine-tuned weights follow the CC BY-SA 3.0 license linked by the upstream model card; this does not assign that license to the entire repository.
