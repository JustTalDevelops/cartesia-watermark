# Development

Run commands from the repository root. The project keeps script entry points in `tools/` so existing research commands and checkpoint provenance remain usable.

## Environments

| Requirements | Purpose |
| --- | --- |
| `requirements.txt` | Main Python 3.13 environment: inference, detector/training, web |
| `requirements/inference.txt` | V3/V2 editing and quality guard |
| `requirements/verification.txt` | Lightweight parent for API-checked reconstruction; model runtime installed separately |
| `requirements/detector.txt` | Local detector and detector training |
| `requirements/web.txt` | FastAPI and Uvicorn; install with inference |
| `requirements/eval.txt` | Supplementary PESQ/DNSMOS evaluation |
| `requirements/research.txt` | AudioSeal and raw-oracle research |
| `requirements/reconstruction.txt` | Isolated Python 3.12 reconstruction environment |
| `requirements/vocoder.txt` | Experimental smoothed BigVGAN reconstruction in Python 3.13 |

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -s tools -p 'test_*.py'
```

The reconstruction runtime is [installed separately](reconstruction.md). FFmpeg/ffprobe are required for the web app and compressed media. API generation and verification need `.env`; local inference does not.

## Source map

| Area | Files in `tools/` |
| --- | --- |
| Audio processing | `reconstruct_audio.py`, `reconstruction_worker.py`, `reconstruction_inference.py`, `verify_reconstruction.py`, `remove_watermark.py`, `remover.py`, `quality_blend.py` |
| Experimental vocoder | `reconstruct_vocoder.py`, `vocoder_inference.py`, `vocoder_assets.json` |
| Experimental filtered prior | `reconstruct_filtered_prior.py`; reuses the isolated reconstruction runtime |
| Upload app | `serve_remover.py`, `check_remover_web.py`; assets in `web/` |
| Detector | `detect_local.py`, `final_detector.py`, `frames.py`, `train_head.py` |
| Training and release | `train_remover.py`, `eval_remover.py`, `export_remover.py`, `verify_remover.py` |
| Supplementary evaluation | `eval_voice_quality.py`, `eval_speaker_quality.py`, `eval_remover_robustness.py` |
| Corpora and API | `cartesia.py`, `corpus*.py`, `sealed.py`, `label_files.py`, `probe.py` |
| Detector reproduction | `frames_contrast.py`, `calibrate_final.py`, `sealed_final.py`, `h2h.py`, `run_r4.sh` |

[experiments/README.md](../experiments/README.md) indexes historical detector probes and reconstruction research. New research scripts live in `experiments/`; reusable runtime code and CLIs live in `tools/`. Avoid moving model implementation files without auditing imports and the frozen source hashes.

## Repository contents

Git includes source, documentation, corpus plans, curated result JSON/text files, three trained detector heads, and small editor release checkpoints with manifests. It excludes generated audio, uploads, feature arrays, environments, caches, logs, intermediate checkpoints, and ZIP exports. See [results/README.md](../results/README.md).

Additional selected trained checkpoints are published as [model release assets](models.md), with an explicit catalog and checksums. Their large binaries stay outside Git history.

Private upload work is stored below `.cache/`; local follow-up research in `experiments/continuous_followup/` stays ignored. No recordings are required for the unit tests. Historical integration/browser checks use generated local fixtures and are separate from the fixture-free test suite.

V3's freeze record contains hashes for its checkpoint and original inference sources. Preserve those files when evaluating runtime changes elsewhere; do not rewrite historical evidence to match a new implementation. The previously built `artifacts/` package is a local snapshot, not the current source tree.

This checkout has no project license selected. Third-party code in `tools/third_party/` includes its upstream license files; reconstruction assets and dependencies have their own licenses.
