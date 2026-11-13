# Experimental V3 editor

V3 is the **Small edits** option in the upload app and the default checkpoint for `tools/remove_watermark.py`. It is a voice-preserving audio editor, not a reliable watermark remover. Subsequent user recordings of roughly one minute still flagged Cartesia's detector after processing. Use [reconstruction](reconstruction.md) for the current processing workflow; its detection results are also mixed.

## Run locally

From the repository root, with the Python 3.13 environment installed:

```bash
uv pip install --python .venv/bin/python -r requirements/inference.txt
.venv/bin/python tools/remove_watermark.py input.wav output.wav
# Smaller edit:
.venv/bin/python tools/remove_watermark.py input.wav output.wav --strength 0.5
# Faster, single-network V2 baseline:
.venv/bin/python tools/remove_watermark.py input.wav output.wav --checkpoint results/remover_quality_v2.pt
```

Inference uses no API key, detector, transformer, or external model download. The CLI preserves sample rate, channel count, and exact sample count; long files use overlapping chunks. Output is FLOAT WAV. It refuses source overwrites and requires `--overwrite` to replace an existing output. `--threads` defaults to 2.

## Model and quality guard

V3 combines two trained waveform networks: 50,210 parameters in a 214 KB checkpoint. Both primarily edit approximately 1–4 kHz, retaining the original high-frequency signal. It does not apply a broad output low-pass filter or EQ.

The guard tries fixed blends with V2. Relative to V2, PESQ may drop by at most 0.015 and ESTOI by at most 0.001. PESQ must remain at least 4.50 when V2 reaches that score; otherwise a candidate cannot lower V2's score. If no blend qualifies, V3 returns V2. The guard checks overlapping 20-second windows and each channel, with a 30 dB global edit-energy budget. These repeated quality measurements contribute to runtime.

The networks were trained on multilingual speech and unmarked controls with voice-disjoint manifests, detector-guided losses, and waveform, spectral, PESQ, and ESTOI penalties. There were no matched pre-watermark targets. V3 bundles V1/V2 weights and a fixed blend policy. Reducing strength or using V2 changes the output and may change detection results.

## Recorded evaluation

The checkpoint and policy were frozen before a test of 16 fresh marked voices (13 English, Hindi, Tamil, Tagalog) and four unmarked controls.

| Full-file fidelity on 16 marked voices | V2 | V3 |
| --- | ---: | ---: |
| Median wideband PESQ | 4.580 | 4.569 |
| Minimum wideband PESQ | 4.427 | 4.427 |
| Median ESTOI | 0.9976 | 0.9974 |
| Median signal-to-edit ratio | 34.8 dB | 34.5 dB |

The largest paired losses were 0.014 PESQ and 0.00063 ESTOI. Three V2 baselines scored below 4.50; V3 returned their exact V2 PCM. These scores compare against marked input, not an unavailable original clean waveform, and do not replace listening.

On a fixed six-file subset originally positive at Cartesia's endpoint, V3 produced 6/6 negative outputs versus 5/6 for V2, with neither of two originally negative controls turning positive. **That small result did not generalize to later real recordings.** Two subsequent roughly one-minute recordings remained detected after V3 processing. An earlier native development pilot also retained detection on two of three files.

The [release report](../results/remover_quality_v3.release.json) records the original evaluation. Supporting reports cover [paired fidelity](../results/remover_quality_v3_testbatch3_paired_quality.json), [actual Cartesia responses](../results/remover_quality_v3_testbatch3_cartesia.json), [DNSMOS](../results/remover_quality_v3_testbatch3_dnsmos.json), and [speaker similarity](../results/remover_quality_v3_testbatch3_speaker.json). The [freeze record](../results/remover_quality_v3.freeze.json) binds the original checkpoint and inference sources. V2 has its own [release report](../results/remover_quality_v2.release.json).

DNSMOS and speaker embeddings are supplementary model estimates, not human quality ratings or identity guarantees. Historical reports refer to generated audio and intermediate artifacts retained locally; these recordings and complete run archives are not bundled in Git.

## Training, evaluation, and export

```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tools/train_remover.py --help
.venv/bin/python tools/eval_remover.py --help
.venv/bin/python tools/export_remover.py --help
.venv/bin/python -m unittest discover -s tools -p 'test_*remover.py'
```

Training and dataset evaluation require regenerated corpora. Supplementary quality tools use `requirements/eval.txt`; raw-oracle experiments use `requirements/research.txt`. `tools/export_remover.py` exports a checkpoint and its manifest, not a self-contained distribution. Previously built ZIPs remain local under `artifacts/`.
