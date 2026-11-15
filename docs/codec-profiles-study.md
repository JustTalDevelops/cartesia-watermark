# Prospective language-profile study

The study is complete: **19/44 marked sources qualified; 25 failed the workflow**.
The [complete report](../results/codec_profiles_reliability.json) retains all 48
requests, including four matched nominal controls. The original frozen codec
study remains 6/14; subsequent use of either cohort is spent-source development.
The procedure below was frozen before this cohort's first TTS request.

The authored plan contains 44 marked requests using 44 public voice IDs absent
from the project's prior scan: four requests in each of eleven languages, with
two Sonic Preview and two Sonic 3.6 requests per language. Four Sonic 3.5 nominal
controls reuse selected marked voices and passages. Two English requests have
predeclared long-duration slots. Four marked requests and the matched English
control use full stereo derivatives. The voice's primary-language tag sometimes
differs from the requested utterance language; both are recorded.

All 44 marked originals tested positive twice. Every selected marked file tested
negative twice and passed all declared criteria, including withheld transcription
reference-error checks. Both long English sources and two of four marked stereo
sources qualified. All eight Hindi/Tamil marked sources failed. These observations
do not establish reliable general removal, unchanged speech or physical erasure.

The four nominal Sonic 3.5 controls had two positive and two negative original
responses. Three processed control outputs qualified and tested negative twice;
one exhausted its routes. Their actual watermark absence is unverified, and
control outputs are not additional marked removal successes. The whole study
retains 96 baseline and 130 candidate-phase HTTP checks with positive source
controls at both ends of selection.

## Fixed processing and evaluation

Only detected language and full duration choose a method order:

| Source | Order |
| --- | --- |
| Default, including short English | Native codec, codec band |
| English longer than 60 seconds | Codec cycle, native codec, codec band |
| French | Codec repair, native codec, codec cycle |
| Russian | Band-prior repair, native codec, codec cycle |
| Korean | Inverse 6 kbps, native codec, codec band |
| Tamil | Band prior, narrow inverse 6 kbps |
| Arabic | Native codec, inverse 6 kbps with bounded word consensus |

The implementation is [codec_profiles.py](../tools/codec_profiles.py). Processing
retains the [prospective recognizer profiles](codec-guard-profiles.md). A source's
original recognition is reused only with the same file and model/source identity.
The generated band prior can be reused across that source's fallbacks. Active
source caches are cleared after each case; models remain warm for the phase.

Speaker similarity of at least 0.95 and a DNSMOS overall delta of at least -0.2
are runtime gates before detector upload. They guide selection and are not
independent validation or human approval. Both processing recognizers must pass.
Reference text and evaluation-ASR scores do not select or repair audio. Source
and plan integrity checks may compare authored metadata, including its text.

All original-only processing, evaluation-ASR and voice calibration must finish
before candidate selection. Hindi/Tamil retain vowel signs and viramas, with a
1% character limit. Other profiles use 3%, with their declared character, word
or Korean spacing policy. Independent transcription evaluation uses Turbo for
Hindi, Large V3 plus MMS for Arabic, and Small elsewhere. Whisper checkpoints
share ancestry. The executable qualification rules are frozen in
[codec_profile_evaluation.py](../experiments/codec_profile_evaluation.py).

At least two original-calibrated checkpoints are required: reference error at
most 0.10 for Hindi/Tamil and 0.15 elsewhere. Every reference-delta criterion
still applies. Wrong-language, rejected, missing, insufficiently calibrated and
failed cases remain in the planned denominator. A selected file failing later
independent speech evaluation remains a failure; that score cannot trigger a
different selection.

## Whole-file detector protocol

Every original is checked twice. Candidate processing begins only after that
baseline and all original-only calibrations finish. Two positive source controls
open and close the candidate phase. Positive candidate responses are confirmed
on the same file; output is published only after two negatives for its exact
encoded bytes. Already checked identical candidate bytes are not queried again.
API errors or conflicting repeats stop the phase without automatic replay.

The direct route uses shared pacing of at least 80 seconds between detector
starts. No source is cropped or padded for detection. Sample rate, natural sample
count and channel count must be preserved. Nominal controls and repeated queries
are separate from positive-to-negative transitions. Neither this protocol nor a
negative result proves physical erasure or universal reliability.

## Shared budget and execution

The existing `.cache/removal-reliability/budget.json` remains the only ledger.
The prepared cohort conservatively adds 72,610 credits, or $4.71965 at the
verified $65-per-million overage ceiling, to the earlier $7.70315 reserve. The
combined reserve is $12.4228 against the authorized $50 cap, not an invoice.
Pricing was checked on November 14, 2026. See [Cartesia pricing](https://www.cartesia.ai/pricing).
There are no paid compute purchases, account upgrades or automatic TTS retries.
Failed or uncertain attempts keep their reservations.

After the source commit, plan and freeze have been published, generation uses
the existing ledger and a separate receipt directory:

```bash
.cache/verification-venv/bin/python experiments/generate_codec_study.py \
  --plan audio/codec_profiles_reliability_plan.json \
  --freeze results/codec_profiles_reliability.freeze.json \
  --directory .cache/removal-reliability \
  --generation-subdirectory codec_profiles_fresh
```

Use `experiments/run_codec_profiles.py --phase materialize` with the authored
plan, then its `evaluation.plan.json` for the `originals`, `baseline` and
`candidates` phases. All three original-only reports are mandatory:
`original_guards.json`, `original_evaluation.json`, and `original_voice.json`.
Transcription evaluation runs through `codec_profile_evaluation.py`; voice scoring
uses `score_codec_study.py --metric voice`. Pass `--originals-only` for the two
original evaluation reports.

Run one neural job at a time. Baseline API checks can run alongside original-only
scoring. Processing/transcription use the pinned Python 3.12 environment; voice
scoring uses the separate Python 3.13 environment. After candidate selection,
score the final files with the same evaluation policies. Raw transcripts and
embeddings remain in ignored local caches; curated reports retain metrics,
hashes, all failures and the actual detector observations.

The final exporter recomputes every attempt's speech assessment from its
file-bound recognition cache, including rejected and detector-positive attempts.
It retains cache digests without exporting recognized text. Skipped recognizers
remain unmeasured; the Arabic consensus path must retain its raw assessment and
the corrections supported by both cached recognizers. This validates the stored
observations without changing the frozen qualification rules.

## Experimental single-file command

The optional wrapper uses the same language/duration profiles and processing
gates, without an authored reference transcript. It uploads the complete original
and qualifying candidates to Cartesia. It requires the pinned codec and voice
environments and assets described in [codec reconstruction](codec-reconstruction.md).

```bash
.cache/resemble-venv/bin/python tools/verify_codec_profiles.py \
  input.wav checked.flac --audit checked.audit.json
```

Set `CARTESIA_API_KEY` in the environment or `.env`. New output and audit paths
are required. The wrapper checks the source positive twice, tries the fixed
profiles, requires the selected encoded file negative twice, and checks the
source positive twice again before publishing the exact checked bytes. Errors,
inconsistent repeats and exhausted profiles produce no output. Its audit retains
the checks and rejection states. Temporary recognition and voice caches are
deleted when the command exits. The web app does not invoke this command.

This wrapper is outside the study's frozen source registry and is not the study
entry point. Its offline tests cover publication boundaries; the active fresh
study evaluates the frozen processing functions. Runtime estimates guide
selection. Without an authored transcript, this command does not perform the
study's independent reference-error evaluation or establish reliable removal.

For Hindi/Tamil, the optional `--experimental-packed-large` flag packs the
Large V3 recognizer's linear weights into FP16 and retains float32 inputs and
outputs. It requires the pinned Python 3.12 x86 CPU environment and uses a
separate recognition-cache identity. Packing is lazy and occurs once per model.
The completed frozen study used the original float32 recognizer.

The [spent-file precision report](../results/codec_precision_development.json)
compares eight files from three originals. Packed FP16 reproduced every token
and all five primary guard decisions. Component generation totaled 236.7 seconds
versus 491.3 seconds for the bound float32 reference. A complete local wrapper
run reproduced the same checked file in 191.4 seconds versus 269.3 seconds, with
real speech/voice gates and eight explicitly mocked detector callbacks. Actual
API requests and pacing add time. These individual CPU runs add no fresh sources
or removal observations and do not establish equivalent behavior on arbitrary
recordings. INT8 changed one guard decision and is retained only in the report.

A separate source-only control reused all nine Hindi/Tamil originals from the
language-profile cohort, including its nominal Tamil control, across eight voice IDs.
Both recognizers' normalized units matched their frozen float32 caches for every
source. Token IDs were not retained in those original caches. This control adds
no candidate/detector observations and leaves the completed study unchanged.

An additional observation-only monitor enforces the protocol's existing stop
rule during the study. The frozen generic verifier can continue after a negative
answer followed by a positive answer and its positive confirmation. The monitor
stops the phase on the first contradiction; the report formatter also rejects
such a sequence. This does not change candidate code, selection order, thresholds
or the planned denominator, and does not authorize replaying requests.
