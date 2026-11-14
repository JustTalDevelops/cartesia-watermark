# Local detector

The round-4 detector reads frozen XLS-R layer-2 features over three-second speech windows, averages three convolutional heads, and compares the mean logit with a threshold set on validation voices. Windows below −50 dBFS are skipped. It needs no Cartesia key.

## Usage

```bash
uv pip install --python .venv/bin/python -r requirements/detector.txt
.venv/bin/python tools/detect_local.py clip.wav other.mp3
```

The first call downloads `facebook/wav2vec2-xls-r-300m` (roughly 1.2 GB). Trained heads `results/head_conv_heldout_x4c{0,1,2}.pt` (about 20 MB total) and `results/final_detector_r4.json` are included. FFmpeg handles formats libsndfile cannot decode.

```python
import os
import sys

os.environ["FEAT"] = "xlsr2"  # Set before importing the detector.
sys.path.insert(0, "tools")
from final_detector import Detector

detector = Detector(config="results/final_detector_r4.json")
score = detector.score(audio, sample_rate)
marked = score is not None and score > detector.threshold
```

`None` means there was no scoreable speech. Run the example from the repository root and supply an audio array and its sample rate.

## Historical results

Ground truth in this investigation is generated provenance: the sampled `sonic-3.6`/`sonic-preview` clips are marked; sampled `sonic-3.5`/`sonic-3` clips are unmarked. These observations describe the models and API version tested in November 2026.

| Test | Local detector | Cartesia endpoint |
| --- | --- | --- |
| sealed4: 192 clips, 3–4.5 s, 48 voices, 14 languages | AUC 0.991; 94% recall, 4% false positives | 18% recall after silence padding to its minimum length |
| Head-to-head: same voices joined into 96 clips, 6–11 s | AUC 0.998; 95.8% accuracy; 92% recall, 0% false positives | 85.4% accuracy; 90% recall, 19% false positives |

The local detector's sealed4 false positives were all `sonic-3.5`. At a stricter 2% false-positive rate, sealed4 recall was 84%, below the original 95% target. See the [full findings](findings.md) for caveats and earlier rounds.

Training used 3,841 original TTS clips from 607 voices in 19 languages, plus augmentations and mark-targeted contrastive edits. Voice-ID splits, multilingual features, and contrastive variants improved generalization. Cartesia's endpoint itself has repeatable false positives and short-clip limitations; it is not clean ground truth. Detector negatives on unusual edited audio may reflect distribution shift.

## Reconstructed speech

The historical accuracy above does not establish accuracy after generative reconstruction. In a targeted diagnostic, an unmarked Spanish control made from `sonic-3.5` and `sonic-3` changed from **−2.263** before full-flow reconstruction to **+2.075** afterward, crossing the fixed **+0.653** threshold. Cartesia's historical ledger flags the original unmarked control too. The [counterexample report](../results/reconstruction_local_shift.json) records hashes, settings, and software versions; the reconstructed control was not queried with Cartesia in this diagnostic.

This demonstrates a local false positive on reconstructed speech, rather than estimating its frequency. Treat scores in this domain as diagnostics. Neither a local positive nor a negative establishes whether the watermark physically remains. The optional [verification CLI](reconstruction.md#check-the-exact-output-with-cartesia) checks the current Cartesia response for the exact output file and records that narrower result.

## Regenerate and train

Copy `.env.example` to `.env` and set a Cartesia key. These commands make paid TTS calls and detector requests; generating the original data used a little over 200,000 TTS characters. Catalogue changes mean a rerun may select different voices than the tracked plans.

```bash
.venv/bin/python tools/corpus.py
.venv/bin/python tools/corpus3.py
.venv/bin/python tools/corpus4.py
.venv/bin/python tools/sealed.py
.venv/bin/python tools/corpus5.py
.venv/bin/python tools/corpus6.py
.venv/bin/python tools/corpus7.py 72000
.venv/bin/python tools/distill_labels.py
```

Feature extraction is resumable and shardable:

```bash
for i in 0 1 2 3 4 5; do
  FEAT=xlsr2 .venv/bin/python tools/frames.py "$i" 6 4 &
done
wait
tools/run_r4.sh
```

`tools/run_r4.sh` builds contrastive variants, trains candidate heads, calibrates on validation, and scores sealed4. It expects `results/sealed4_cartesia.json`; `tools/label_files.py` can create a ledger for regenerated clips. Training needs roughly 25 GB RAM per head; the script trains sequentially.

All four sealed sets have already influenced the research. A new model comparison requires fresh voices and sentences. API detector requests are deliberately paced, serialized, and resumable; do not treat those waits as inference bottlenecks.
