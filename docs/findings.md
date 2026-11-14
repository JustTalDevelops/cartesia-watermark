# Cartesia audio watermark — findings

Historical black-box probes from November 11–12, 2026, using generated Sonic TTS audio and a paid API key.
The observations below come from the public API and Cartesia's detector
(`POST /v1/watermark/detect`, which returned only `{"detected": bool}`). At the time, Cartesia's watermarking
guide described the feature as always on, without identifying its technique.

**These probes do not establish reliable removal or reveal the implementation.** The later
[frozen codec study](codec-study.md) qualified only 6/14 marked sources. See the
[current project status](../README.md) and [development follow-ups](codec-development.md).

Historical negative labels use requested older-model provenance, principally `sonic-3.5` and `sonic-3`;
their actual watermark absence was not independently verified. Accuracy and false-positive rates below
are measured against those assigned labels. A positive reply on a nominal control does not by itself
establish a detector false positive, and a negative reply does not prove watermark erasure. Frequency,
mechanism and pipeline descriptions remain hypotheses from a limited set of probes.

## Main observations

- **Model coverage in these probes:** detector-positive output came from **fluent-bird** (`sonic-3.6`,
  `sonic-3.6-2026-08-27`) and **wise-puddle** (`sonic-preview`, `sonic-latest`, `sonic-3-latest`).
  Older requested models supplied nominal negative labels; some `sonic-3.5` outputs also tested positive.
  Shared detection across two model families does not locate watermarking before, within or after synthesis.
- **Audio evidence:** detection survived tested container and encoding changes. Matching WAV headers alone
  do not identify the mechanism.
- **Tested serving paths:** `/tts/bytes`, `/tts/sse` and the WebSocket API returned detectable examples.
- **Tested formats:** detectable examples included 8, 16, 22.05, 24, 44.1 and 48 kHz, f32/s16/μ-law,
  and MP3 at 32 and 128 kbps. Their spectra suggest a 44.1 kHz source rate; these observations do not
  establish where watermarking occurs relative to resampling and encoding.
- **Frequency probes:** the **1.5–3.5 kHz** band alone tested positive on some recordings; replacing it
  changed those detector answers to negative. This does not prove removal or a universal frequency boundary.
- **Mechanism hypotheses:** tested spectral and phase edits suggest signal-dependent features concentrated
  around 2–3.5 kHz. Marked-provenance models had about +1 dB more energy there, but different synthesis
  models confound that comparison. No shared fixed carrier, comb, echo or symbol rate was resolved in
  these analyses. Mixing, splicing and band delays suggest temporal integration; they do not reveal the
  detector architecture. Stock AudioSeal/WavMark probes did not identify a matching watermark.
- **Stayed detectable in tested examples:** gain, polarity flip, sample shifts, MP3 32k/64k, Opus 16k, resampling to 8 and 16 kHz,
  ±5% speed, noise at 20 dB SNR, and splicing (25% marked audio is enough).
- **Tested negative after:** time reversal, low-pass ≤~2.7 kHz, high-pass ≥~1.6 kHz, white noise ≤14 dB SNR,
  delaying the 2–3.5 kHz band by ≥33 ms, mixing in ≥~1/3 unmarked speech, full-band phase randomisation.
- **Local detector: works, including on short clips and unseen languages.** `tools/detect_local.py` (round 4:
  XLS-R-300M layer-2 features over 3 s windows, three conv heads trained with mark-targeted contrastive variants).
  On a fresh sealed set (48 unseen voices, 14 languages, 2/3 non-English, Tamil and Tagalog never seen in
  training, clips 3–4.5 s), scored once: **AUC 0.991, catches 94% of marked clips at a 4% false-positive rate**
  (84% at 2%, 98% at 10%). All of its false positives were `sonic-3.5`. **Head-to-head on the same voices joined
  into ~8 s clips: it beats Cartesia's own detector**, 95.8% vs 85.4% accuracy (0% vs 19% false positives, 92% vs 90%
  caught). See §9 for the four rounds.
- **Cartesia's detector is not a clean ground truth:** it rejected audio shorter than 5 s, and silence-padded
  3–4 s clips had only 18% recall in the tested cohort. It flagged 13–30% of unedited `sonic-3.5` clips in
  several cohorts despite their assigned negative labels. See §8.
- **Caveat on negative replies:** reversed, phase-scrambled or overlapping speech may be outside the
  detector's useful input distribution. Neither positive nor negative replies prove watermark presence or
  absence; repeated source controls help interpret the measured transition.
- **Some unexpected positives repeated:** an unedited `sonic-3.5` clip, a **pure 1 kHz sine tone**,
  3.5 audio low-passed at 8 kHz, and the last 6 s of a 3.5 clip were flagged. The sine-tone result is a
  test-signal counterexample; older-model results retain the provenance limitation above. Repeated answers
  on these files do not establish universal determinism or a population false-positive rate.

## Setup

- API version `2026-08-14`. TTS through `POST /tts/bytes`, defaulting to WAV f32 at 44.1 kHz with voice "Daniel".
- Detector: `POST /v1/watermark/detect` (multipart `file`). It requires clips of **at least 5 s**; shorter clips get a 400.
- Corpora:
  - `audio/corpus/`: 4 texts × 2 voices × {3.6, 3.5}
  - `audio/corpus2/`: 12 texts × 4 voices × {3.6, preview, 3.5, 3}
  - `audio/fmt/`: one text in every output format
  - Detector training and testing (§9): `corpus3`–`corpus7` and the sealed test sets `sealed`, `sealed2`–`sealed4`,
    4 models per voice and sentence; voice/text plans are tracked as `audio/*_plan.json` and `audio/*/voices.json`
- Detector results are cached by file hash in `results/detect_cache.json`. Phase-2 results are in `results/phase2.json`.

## 1. Model coverage

| Public model ID | Internal model | Detected |
|---|---|---|
| sonic-3.6 | fluent-bird | yes |
| sonic-3.6-2026-08-27 | fluent-bird | yes |
| sonic-preview | wise-puddle | yes |
| sonic-latest | wise-puddle | yes |
| sonic-3-latest | wise-puddle | yes |
| sonic-3.5 | azure-disco | mostly no; one repeated positive in this probe, see §6 |
| sonic-3 | ? | no |
| sonic-2 | n/a | n/a (400 `model_sunsetted`) |

Internal names come from the `/infill/bytes` error message ("model fluent-bird does not support infill").
None of the watermarked models support infill, so there's no way to get our own audio back with the mark applied.

Side observation: every model's output lies exactly on the **float16** grid. wise-puddle output also lies
on the **bfloat16** grid, while fluent-bird's doesn't, which fits them being different models.

## 2. Output formats (all sonic-3.6, all detected)

8000 / 16000 / 22050 / 24000 / 44100 / 48000 Hz WAV, pcm_s16le, pcm_mulaw at 8 kHz, MP3 at 128 kbps / 44.1 kHz,
MP3 at 32 kbps / 22.05 kHz. The 48 kHz output has a hard cutoff at 22 kHz, which shows the model's native rate is 44.1 kHz.

## 3. Robustness battery (sonic-3.6 clip `t0_daniel`, with its sonic-3.5 twin as control)

| Transform | 3.6 | 3.5 control |
|---|---|---|
| none | **T** | F |
| gain −20 dB | **T** | F |
| polarity invert | **T** | F |
| **time reverse** | F | F |
| first 5 s | **T** | F |
| last 6 s | **T** | **T** (false positive, repeatable) |
| shift by 1 or 257 samples | **T** | F |
| low-pass 8 kHz | **T** | **T** (false positive, repeatable) |
| low-pass 4 kHz | **T** | F |
| low-pass 2 kHz / 1 kHz | F | F |
| high-pass 1 kHz | **T** | F |
| high-pass 4 kHz / 8 kHz | F | F |
| resample via 16 kHz | **T** | F |
| resample via 8 kHz | **T** | **T** (false positive) |
| MP3 64k / 32k | **T** | F |
| Opus 16k | **T** | F |
| white noise SNR 20 dB | **T** | F |
| white noise SNR 10 / 0 dB | F | F |
| speed ×1.05 (resample) | **T** | F |

Repeated queries agreed on the tested files. This does not establish behavior for other files or future
service versions; conflicting responses must remain a verification failure.

## 4. Band localisation (Butterworth band-pass and band-swap, 3.6 vs 3.5 twin)

| Band | Only 3.6's band | 3.5 + 3.6's band | 3.6 with 3.5's band |
|---|---|---|---|
| 2–4 kHz | F | F | F |
| 1–4 kHz | **T** | F | F |
| 1.5–3.5 kHz | **T** | F | F |

- In these examples, 1.5–3.5 kHz alone tested positive and swapping it out changed the reply to negative.
  This is an observation about detector behavior on these files, not a necessity/sufficiency proof for the watermark.
- 2–4 kHz alone is not enough, so the 1.5–2 kHz part matters.
- The tested band swaps into nominal older-model speech did not produce detection, while a swap into
  another marked-provenance clip did (§7). Mixing may affect detection; these tests do not rule out
  host-dependent behavior or establish the classifier's decision rule.

## 5. Temporal behaviour

- Splices of "3.6 for the first X%, then 3.5" are detected for X = 75, 50 and **25** (~3 s of marked audio in 12 s).
- Every 5 s sliding window of a 3.6 clip is detected on its own, so the mark is present throughout.
- Time reversal destroys it. Polarity inversion doesn't. Full-band phase randomisation destroys it while
  blurring the magnitude doesn't (§7), so the evidence is in the waveform's fine structure, read directionally.
- `<break>` pauses come out as exact digital zeros in both 3.5 and 3.6, so no mark is added to silence.
  The mark lives only in generated speech.

## 6. Repeated positives on nominal controls and test signals

| Input | Detected |
|---|---|
| unedited sonic-3.5 clip ("library" text, `audio/fmt/ref_3.5.wav`) | **T** |
| pure 1 kHz sine, 8 s | **T** |
| sonic-3.5 low-passed at 8 kHz | **T** |
| last 6 s of a sonic-3.5 clip | **T** |
| other unedited sonic-3.5 clips (8) | F |
| digital silence 6 s, white noise, low-passed noise | F |

## 7. Mechanism probes (Cartesia detector, sonic-3.6 clips A/B/C)

| Probe | Result |
|---|---|
| Randomise STFT phase, full band | **F** (2/2) |
| Randomise phase in 1.5–3.5 kHz only | T on A, F on B |
| Blur magnitude over time in 1.5–3.5 kHz (phase kept) | **T** (2/2) |
| Any single 250 Hz notch between 1 and 4.5 kHz | **T** (14/14) |
| Only the 1.5–3.5 kHz band | **T** (3/3) |
| 1.5–3.5 kHz band replaced by the 3.5 twin's | **F** (3/3) |
| 1.5–3.5 kHz band replaced by another *marked* clip's | **T** |
| 1.5–3.5 kHz band delayed 1 / 5 / 20 ms | **T** |
| 1.5–3.5 kHz band delayed 33–100 ms | **F** (5/5 incl. distillation labels) |
| First 0.5 / 1 / 2 s marked, rest unmarked twin | F / F / **T** |
| White noise at 12 / 14 / 16 / 18 dB SNR | F / F / **T** / **T** |
| 68% marked + 32% unmarked twin (sample mix) | **F** |

Interpretation: these results are consistent with sensitivity to fine waveform structure in this band and
temporal context around 1–2 seconds. The tested files tolerated about 20 ms of band misalignment, and mixing
nominal older-model speech changed some answers. This interpretation does not identify the watermark or detector.

Offline analyses on 720 clips (30 voices × 6 texts × 4 models):
- **Energy footprint:** both marked models sit +0.5 to +1.1 dB above both unmarked models in every 250 Hz band
  from 2.0 to 3.5 kHz (all four pairings agree; same-voice 3.6 > 3.5 twin in 70–81% of pairs).
- **No echo hiding:** cepstral differences are smooth band-shape ripples, not a delay spike.
- **No fixed carrier or symbol rate:** the cyclostationary spectrum of the 2–3.5 kHz band shows no line shared by the
  marked models (max whitened peak ~1.5×, at noise level).
- **AudioSeal / WavMark probes:** AudioSeal's public detector scored about 0.005 on all four models and
  WavMark found no payload. Failed recognition by those tools does not rule out related techniques.
- **No matched clean target was obtained:** the tested bytes, SSE and WebSocket outputs were detectable,
  and the tested marked-provenance models rejected `/infill`. These probes did not provide the same
  generated waveform both before and after watermarking.

## 8. How good is Cartesia's own detector?

The repeated positive sine-tone observation in §6 shows why a detector reply alone is insufficient source
attribution. Older-model positives and their edited variants also conflict with assigned negative labels,
but do not independently establish watermark absence. Neither repeated agreement nor these label conflicts
establishes universal detector determinism.

Measured against provenance on clean TTS clips (truth = which model made the clip):

| Set | Marked caught | sonic-3.5 flagged | sonic-3 flagged |
|---|---|---|---|
| corpus3 held-out voices, long English clips (6–7 s) | 24/24 | 6/12 | 0/12 |
| corpus5 validation voices, 4–6 s, 8 languages | 63/76 (83%) | 11/38 (29%) | 1/38 |
| sealed2, 4–6 s, 13 languages | 53/60 (88%) | 8/30 | 0/30 |
| sealed3, 4–6 s, 12 languages | 66/80 (82.5%) | 11/40 | 0/40 |
| sealed4, **3–4.5 s (padded to 5.2 s with silence)** | **17/96 (18%)** | 0/48 | 0/48 |

- **Duration dominates its recall.** The API rejects clips under 5 s, so short clips must be padded, and padding
  hurts it. On sealed3 it caught 65% of 4–5 s clips and 100% of clips over 5 s; on sealed4 (median 3.9 s), 8% of
  clips under 4 s. This fits the mixing result in §7: unmarked (here: silent) audio counts against the mark.
- **Nominal sonic-3.5 controls often tested positive:** roughly a quarter to a third in several cohorts.
  Local scores predicted those answers poorly (AUC 0.67). This does not distinguish detector error,
  shared synthesis traits, actual watermarking or another backend cause.

## 9. Local detector: attempts and status

The core obstacle: every unmarked clip we can obtain comes from a *different model* than every marked clip, so a
learner can separate the classes by model timbre without ever finding the mark. Each attempt and how it was caught:

| Version | Approach | Result |
|---|---|---|
| v1 | logistic regression on band spectra | leave-model-out 36%: pure timbre |
| v1b | small STFT-magnitude CNNs | 0.50–0.60 on unseen models |
| v2 | complex-STFT CNN, 1–4.5 kHz, 720 clips | unseen models 0.60; flagged white noise as marked (learned the energy footprint) |
| v3 | raw-waveform CNN + noise negatives | AUC 0.84 on unseen voices, but sonic-3.5 flagged 50%: still timbre |
| v4/v5 | waveform CNN on rule-labelled transforms | AUC 0.71–0.74; learned "processed audio → negative" |
| **WavLM clip-level** | pretrained WavLM layer-1 features + logistic regression | **leave-model-out AUC 0.91 / 0.90 (both directions)**; 0.853 on the probe set |
| WavLM band-limited | same, input restricted to 1–4.5 kHz | worse (probe 0.82): Cartesia's detector itself uses out-of-band speech |
| WavLM windowed (round 1) | layer-1 features over 3 s windows, noise and reversal augmentation, mean pooling, threshold 0.74 | see below |

### Rounds 1–4 (each scored once on its own fresh sealed set of unseen voices)

| Round | Features and head | Training data | Sealed result |
|---|---|---|---|
| 1 | WavLM layer 1, 3 s windows, logistic regression | ~1,300 clips, 26–30 voices | sealed: AUC 0.962, TPR 73%, FPR 2.9% (en/es/fr) |
| 2 | WavLM layer 1, conv heads + fine-tuned WavLM, ensemble | +330 voices (corpus5), 8 languages | sealed2: AUC 0.949, TPR 78%, FPR 5% |
| 3 | WavLM layer 1, conv heads + mark-targeted contrastive variants | +124 voices (corpus6), sealed sets 1–2 | sealed3: AUC 0.945, TPR 70%, FPR 5% |
| **4** | **XLS-R-300M layer 2**, conv heads + contrastive variants | + sealed3 | **sealed4: AUC 0.991, TPR 94%, FPR 4%** |

The sealed sets differ (sealed4 has shorter clips and more languages), so rows aren't strictly comparable. Each
was still a one-shot test on voices, and for sealed2–4 sentences, that no training or model choice had seen.

What mattered, in order:
1. **Multilingual features (round 4).** Rounds 2–3 sat at ~0.996 AUC on English validation voices but ~0.945 on
   non-English, and the sealed sets are mostly non-English. WavLM is pretrained on English. A linear probe on
   XLS-R (pretrained on 128 languages) beat the whole WavLM ensemble (validation AUC 0.993 vs 0.986), and the
   round-4 heads reached 0.998 English / 0.988 non-English on validation.
2. **Band-focused contrastive variants (rounds 3–4).** Each marked-provenance training clip had three band
   edits intended to reduce model-timbre confounding. Labels followed Cartesia's replies to those edits (§7):
   10 ms delay → detected; 40 ms delay → not detected; replacement from the same voice and sentence requested
   through sonic-3.5 → not detected. Agreement with Cartesia on mixed audio absent from training rose from
   0.58 to 0.68, and the variants won the validation comparison in both rounds. This does not prove that the
   local detector learned the watermark itself.
3. **Voice-disjoint splits.** Cartesia lists multilingual voices under several languages, so per-language sampling
   picked the same voice more than once; 6 sealed2 voices had leaked into training before splits went by voice ID.
4. More voices alone (round 3: +30%) did **not** help.

Round-4 detector on sealed4 (`results/sealed4_final.json`): per model, 96% of sonic-3.6 and 92% of sonic-preview
caught, 8% of sonic-3.5 and 0% of sonic-3 flagged. All 14 languages at ≥4/6 marked caught except Portuguese (2/4).
Tamil and Tagalog (never seen in training): 11/12 caught, 0 false positives.

**Head-to-head with Cartesia's detector on unpadded audio** (`tools/h2h.py`, `results/h2h4_final.json`). Each
sealed4 voice's two marked clips (sonic-3.6 + sonic-preview) were joined into one ~8 s clip, and its two unmarked
clips (sonic-3.5 + sonic-3) into another: 96 clips of 5.8–11 s, no padding, both detectors on identical audio,
nothing tuned on it.

| 96 joined clips (48 marked) | Marked caught | Unmarked flagged | Accuracy |
|---|---|---|---|
| **Local detector (round 4)** | 44/48 (91.7%) | **0/48 (0%)** | **95.8%** (AUC 0.998) |
| Cartesia `/v1/watermark/detect` | 43/48 (89.6%) | 9/48 (18.8%) | 85.4% |

Its four misses sit just under the threshold (it was set on single short clips); Cartesia's misses are on
different clips. Cartesia's nine false positives include both Tamil unmarked clips.

Honest limits:
- At the validation-chosen threshold it is 94% / 4%, short of the 95%-at-2% target I set. Threshold-free on
  sealed4: 84% caught at 2% FPR.
- Its remaining false positives are sonic-3.5, the same model Cartesia's detector struggles with.
- Validation (360 clips) is ~73% English, so it is optimistic. Judge on the per-language numbers.
- It can't fully rule out that part of the signal is a vocoder trait shared by fluent-bird and wise-puddle rather
  than the mark; the contrastive variants and its agreement with Cartesia on edited audio argue against that.
- All four sealed sets are now spent. A further iteration needs new TTS audio for a fresh test.

## Open questions

- What the detector actually computes (score, threshold, pooling). The mix and splice results suggest per-window
  evidence for *and against*, not "any window fires".
- The generator itself. Without it (or same-model unmarked audio) a local detector must be learned indirectly.
- Whether a payload is present. Cartesia says it currently identifies only "Cartesia-generated", with
  per-customer attribution "planned". If it lands, comparing clips across two API keys would show it.
- Speed range, codec bitrate floor.
- Whether it survives time-stretching without pitch change or re-recording through a speaker and microphone.
  Later [codec experiments](codec-development.md) explored EnCodec/DAC and retain both successes and failures.

## Detector rate limit (operational note)

During these historical probes, the detector returned empty-body 429 responses even without an API key,
after roughly 12–15 calls in about two minutes from one IP; some pauses exceeded ten minutes. TTS and
`/voices` did not show the same behavior. This observation does not identify the enforcing backend or a
stable service quota. Historical exploration harnesses cache answers and may retry; the later
[frozen study protocol](codec-study.md) uses direct shared pacing of at least 80 seconds and stops on API
errors or inconsistent repeats without automatic replay.

## Reproduce

Setup, data generation, training and evaluation of the current detector: see [detector setup](detector.md).
The mechanism probes in §1–§7 are in [`experiments/`](../experiments/README.md), for example:

```
.venv/bin/python tools/corpus.py && .venv/bin/python experiments/corpus2.py
.venv/bin/python experiments/battery.py audio/corpus/t0_daniel_sonic-3.6.wav audio/corpus/t0_daniel_sonic-3.5.wav
DETECT_INTERVAL=45 .venv/bin/python experiments/phase2.py      # resumable, results in results/phase2.json
.venv/bin/python experiments/mechanism.py && .venv/bin/python experiments/align.py
```
