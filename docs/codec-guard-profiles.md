# Prospective codec guard profiles

This development branch changes speech assessment. It has no new held-out
removal result. The frozen V3 waveform editor and the original codec study keep
their recorded results. Previously used sources remain development inputs.

## Recognizers and comparison units

| Utterance language | Primary recognizer | Second recognizer | Paired comparison limit |
| --- | --- | --- | --- |
| Hindi, Tamil | Whisper Large V3 FP32 | MMS FP32 | 1% characters, including essential vowel signs and viramas |
| Arabic | Whisper Turbo FP32 | FastConformer Arabic CTC INT8 export | 3% words |
| English | Whisper Turbo FP32 | SenseVoice FP32 | 3% words |
| Japanese, Chinese | Whisper Turbo FP32 | SenseVoice FP32 | 3% characters |
| Korean | Whisper Turbo FP32 | SenseVoice FP32 | 3% spacing-normalized words |
| Spanish, French, German, Russian | Whisper Turbo FP32 | MMS FP32 | 3% words |

Whisper Turbo identifies the original language. Hindi/Tamil then use the larger
checkpoint for transcription. Candidates force the same language and recognizer
profile. The larger checkpoint adds CPU work; measured runtime differences are
case-specific.
Both India-language comparisons retain combining marks. Arabic comparison removes
vowel marks and tatweel without merging distinct letters. Chinese writing systems
use pinned OpenCC conversion; Korean spacing follows the existing diagnostic.

Character-error and word-error percentages measure different things. The new
one-percent character limit cannot be substituted into the historical study or
claimed equivalent to its three-percent word limit. A future validation protocol
must specify reference scoring, source calibration and evaluation-only checkpoints
before generating new test recordings. Raw word-error diagnostics should also be
retained. The separate [language-profile protocol](codec-profiles-study.md)
specifies these roles and limits before its new generation phase. Its freeze is
not efficacy evidence.

## Assets and licenses

Install the [V2 dependencies and models](codec-v2-development.md), then the two
additional entries in [`tools/codec_assets.json`](../tools/codec_assets.json):

- `large`: the exact local files from `openai/whisper-large-v3` at the recorded
  revision, under Apache-2.0. It shares Whisper ancestry with Turbo.
- `fastconformer`: the exact ONNX graph and token file from
  [`krut42/voice-fastconformer-ar-ctc-int8`](https://huggingface.co/krut42/voice-fastconformer-ar-ctc-int8)
  at the recorded revision. This is a community INT8 conversion derived from
  [NVIDIA's Arabic checkpoint](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0),
  under CC-BY-4.0. Numerical equivalence to the original FP32 model is not claimed.

The manifests pin every loaded file. These upstream models were not trained by
this project and are not included in its trained-model release. Processing makes
no model downloads. Run `tools/reconstruct_codec.py --check` in the pinned Python
3.12 environment after installation.

The `codec-6k` preset reconstructs the original at the pinned checkpoint's
6 kbps setting. It does not re-encode the first 12 kbps pass. The final whole
PCM24 file still needs both speech guards and separate detector verification.
The three-percent default and one-percent Hindi/Tamil limits also apply here.

The original profile implementation passed 173 tests. Later method and inference
changes passed 97 targeted checks. Actual whole-file integrations and source-only
calibration are recorded in the [development reports](codec-development.md),
including a [saved-file compatibility replay](../results/codec_profile_replay.json).
These reuse spent recordings and add no held-out source. A local speech check
or detector-negative response does not establish reliable erasure.
