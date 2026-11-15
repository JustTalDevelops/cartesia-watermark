# Codec development results

The original [fresh study](codec-study.md) remains **6 qualified marked sources out
of 14**. The development follow-ups below reuse those recordings. A later fresh
cohort is reported under [Completed language-profile study](#completed-language-profile-study).
Neither study met the reliability target.

| Follow-up | Observed result |
| --- | --- |
| [French local repair](../results/codec_repair_development.json) | The exact repaired output passed seven estimated quality gates and returned two negative detector responses. |
| [Long English recording](../results/codec_cycles_development.json) | A second native-codec pass preserved the whole 101.12-second recording, passed seven estimated quality gates, and returned two negative responses. |
| [Korean lower bitrates](../results/codec_bitrate_development.json) | Both 3 and 6 kbps band reconstructions passed seven estimated quality gates but stayed detectable twice. The speech-rejected native 3 kbps output was not queried. |
| [Korean inverse mixture](../results/codec_followup_development.json) | Generated native 6 kbps speech outside 1–4 kHz, with the generated band prior inside, passed seven quality estimates and cleared twice with positive source brackets. The narrower mixture and timing probes failed quality. |
| [Russian band-prior repair](../results/codec_followup_development.json) | A half-second generated-donor patch repaired the failed guard. The complete output passed seven quality estimates and cleared twice with positive source brackets. |
| [Arabic inverse mixture and consensus](../results/codec_followup_development.json) | The explicit consensus profile retained the raw rejection, discounted one jointly supported transcription correction, and passed the other independent checks. The exact file cleared twice with four positive source controls. |
| [Tamil gentle vocoder](../results/vocoder_codec_tradeoffs.json) | Both recordings passed the prospective quality checks but stayed detectable twice. |
| [Tamil stronger vocoder](../results/vocoder_codec_tradeoffs.json) | Midpoint and stronger smoothing cleared detection on the first recording but increased independent recognizer errors. Stronger smoothing also reduced estimated naturalness. These are failed removals. |
| [Tamil band-focused vocoder](../results/vocoder_band_development.json) | Smoothing focused on 1–4 kHz passed seven estimated quality checks on the first recording and returned two negative responses with four positive source controls. The stronger variant failed naturalness and was not queried. This is one spent source. |
| [Tamil separate smoothing and repair](../results/vocoder_codec_tradeoffs.json) | Separate frequency/time smoothing failed the speech guards. A bounded repair passed the processing guards but failed independent speech and naturalness checks. None was detector-tested. |
| [Second Tamil original-band mixture](../results/codec_followup_development.json) | Keeping original audio outside the generated middle band passed quality but remained detectable twice. Stronger smoothing failed independent speech evaluation and was not queried. Neither mixture establishes physical erasure. |
| [Second Tamil follow-up](../results/codec_ta2_followup.json) | The generated inverse mixture narrowed to 1.5–3.5 kHz passed seven estimated quality criteria and cleared twice with four positive source controls. Earlier split-band, Mimi, repair, and waveform-reconstruction failures remain recorded. |
| [Arabic Mimi codec](../results/vocoder_codec_tradeoffs.json) | The 2.2 kbps output passed the measured quality checks but stayed detectable twice. The 1.1 kbps output failed speech checks. The 1.65 and 1.925 kbps outputs passed seven estimates but failed an additional MMS check. The rejected files were not detector-tested. |

The prospective language profiles differ from the frozen study. Tamil uses
Whisper Large V3 and MMS guards with a one-percent character limit that preserves
vowel marks; Whisper Small provides an additional evaluation. Arabic uses Turbo
and a pinned FastConformer export with a three-percent word limit; Large V3 and
an additional MMS review provide evaluation. Both middle-bitrate Mimi outputs'
MMS reference error increased by 0.0423, exceeding 0.03 despite the other passing
estimates. Checkpoint roles, calibration, hashes, voice
estimates, every planned rejection, and complete detector controls are retained
in the reports. Whisper checkpoints share ancestry.

The [experimental band-prior branch](https://github.com/JustTalDevelops/cartesia-watermark/tree/codec-band-prior-20261114)
packages the first Tamil result as `codec-band-prior`. An actual run with both
recognizers reproduced the tested FLAC bytes exactly; the report binds its source
commit. Processing and worker shutdown took 205.2 seconds for 30.08 seconds of
audio on the four-core CPU machine, excluding initial outer-runtime construction.
It adds no fresh validation and does not change the app's methods.

The same branch also includes `codec-band-prior-repair` and `codec-band-inverse6`.
Both packaged paths reproduced their detector-checked Russian/Korean files with
real guards. Processing and worker shutdown took 148.1 seconds for 32.88 seconds
of Russian audio and 60.8 seconds for 29.92 seconds of Korean audio, excluding
initial outer-runtime construction. These are single-case timings, with different
recognizers; they do not measure an app-wide speedup. The [follow-up report](../results/codec_followup_development.json)
retains the unsuccessful Arabic and second Tamil band-prior outputs, the rejected
second codec passes, and both discarded timing variants.

The opt-in Arabic consensus path records both the raw 3/71 disagreement and the
adjusted 2/71 comparison. A correction needs one unique phrase with two unchanged
context words on each side in the other recognizer's original and processed
transcripts; at most two corrections and 3% of source words are allowed. The
comparison limit remains 0.03. Reference text and detector labels do not enter
this rule. Large V3, MMS, speaker, and naturalness checks remain separate. The
packaged path reproduced the checked file using real models in 91.6 seconds for
31.68 seconds of audio, excluding initial outer-runtime construction. The raw
failure is not reclassified as a pass under the original profile.

A [compatibility replay](../results/codec_profile_replay.json) reassessed nine
previously selected whole files with the prospective language profiles. All nine
passed the current automated quality estimates: seven marked files and two
nominal controls. This reused the exact previously detector-checked bytes, with
no new synthesis, HTTP requests, or independent sources. Hindi passed the new
vowel-preserving character policy with Turbo evaluation; its historical failure
under the original policy remains recorded. The frozen fresh-study result stays
6/14, and the two controls are not removal transitions.

The second Tamil [follow-up](../results/codec_ta2_followup.json) retains eleven
rendered variants and two proposals rejected before rendering. The
short repair covered 0.68 seconds of the 26.48-second recording. Although its
4.4 kbps Mimi output passed both processing guards, independent Small character
error increased by 0.0215, exceeding 0.01. The unpatched output passed independent
quality checks but failed its primary processing guard. Neither was detector
tested. A separate sixteen-character-context consensus review found no supported
isolated substitutions and changed no guard decision. The joint lower-bitrate
repair would cover about 20.3%, exceeding the unchanged 20% ceiling.

Two additional forward/inverse waveform reconstructions failed speech criteria
and were not detector tested. A numerical repair follow-up preserved untouched
PCM24 samples exactly and reproduced its identical-donor control byte for byte.
Its independent speech error still increased by 0.0215, so it remains a failed
removal. The alternative generated donor failed the unique-context safety rule
before rendering. No speech limit or reference-based selection was changed.

The subsequent fixed 1.5–3.5 kHz inverse mixture passed both processing guards
and the independent speech, speaker, and naturalness estimates. Its exact whole
file returned two negatives between four positive original controls. The
packaged `codec-band-inverse6-narrow` path reproduced the same FLAC bytes using
real recognizers in 178.6 seconds for 26.48 seconds of audio, excluding outer
runtime construction. This is another treatment of the same spent recording,
with zero fresh independent sources; it does not change the frozen 6/14 study.

These are automated estimates, without human listening approval. A negative
detector response does not establish physical erasure. Quality failures remain
failures even when the detector returns negative twice.

The experimental branch's [inference follow-up](../results/codec_inference_followup.json)
reproduced all four checked files after deferring unused codec passes. Those
passes cost less than a second each, so their removal did not establish a total
speedup. A second change runs the secondary recognizer first and skips the primary
only after a rejection; accepted outputs still need both. One warmed rejection
took 26.6 seconds before and 10.2 seconds after. Complete Russian repair took
141.7 versus 131.4 seconds, while the accepted Korean path took 61.6 versus 66.0
seconds. These individual CPU runs exclude outer-runtime construction and do not
establish an app-wide speedup. Both final files reproduced the checked bytes.

A further actual fallback run reused one original recognition and one generated
band prior. Original recognition took 76.2 seconds, the first method 104.7
seconds, and the cached narrow fallback 66.8 seconds. Both outputs reproduced
their checked bytes. These scopes differ from standalone processing; original
calibration is not free. Current-profile French repair and long-English cycle
runs also reproduced their previously checked files. The new operational voice
gate accepted the narrow Tamil output and rejected a known naturalness failure.
All of these are spent-source preflights.

An offline integration of the experimental single-file profile wrapper reproduced
the checked narrow Tamil file using real speech and voice models. It reused one
original transcription and one generated prior across both methods. The complete
local workflow took 269.3 seconds, including model construction and original
calibration, with eight explicitly mocked detector callbacks and no HTTP calls.
This verifies packaging and exact-file parity; it adds no removal observations.
The earlier failed call-counter instrumentation attempt remains in the report.

The separate [precision comparison](../results/codec_precision_development.json)
uses eight files from three spent originals. INT8 reduced Large V3 generation
time but changed one primary guard decision. Packed FP16 matched every token and
all five primary decisions; generation totaled 236.7 versus 491.3 seconds for the
bound float32 reference. A real full-wrapper integration reproduced the same
checked file in 191.4 versus 269.3 seconds, about 29% less local time. Both wrapper
runs used mocked detector callbacks; actual API pacing adds time. The
[experimental flag](https://github.com/JustTalDevelops/cartesia-watermark/blob/codec-band-prior-20261114/docs/codec-profiles-study.md#experimental-single-file-command)
has its own cache identity and was excluded from the completed float32 study.
These individual CPU observations add no fresh sources or removal evidence.

An additional source-only control reused all nine Hindi/Tamil originals from the
language-profile cohort, across eight voice IDs. Both recognizers' normalized units matched
their frozen float32 caches for all nine. It read no candidate outputs and adds
no detector observations or paired float32 timing comparison.

## Completed language-profile study

The separate [prospectively frozen study](https://github.com/JustTalDevelops/cartesia-watermark/blob/codec-band-prior-20261114/docs/codec-profiles-study.md)
qualified **19/44 marked sources**; 25 failed the workflow. The
[complete 48-request report](../results/codec_profiles_reliability.json) retains
every marked failure, four matched nominal controls and all detector receipts.
The original 6/14 result remains unchanged.

There were four marked requests per language, two Sonic Preview and two Sonic 3.6,
using 44 public voice IDs absent from the project's prior scan. Two English
recordings exceeded 100 seconds; four marked recordings used stereo derivatives.
This does not establish unseen training data for upstream models or coverage of
arbitrary recordings. The small language groups are not population-rate estimates.

| Language | Qualified / planned marked sources |
| --- | ---: |
| English | 3/4 |
| Spanish | 3/4 |
| French | 2/4 |
| German | 2/4 |
| Hindi | 0/4 |
| Japanese | 3/4 |
| Arabic | 2/4 |
| Korean | 2/4 |
| Mandarin Chinese | 1/4 |
| Russian | 1/4 |
| Tamil | 0/4 |

All 44 marked originals tested positive twice. The 19 selected marked outputs
tested negative twice for their exact whole-file bytes and passed the withheld
reference-error checks and remaining declared criteria. The study recorded 96
baseline and 130 candidate-phase HTTP checks, including repeated opening and
closing positive source controls. Source sample rate, natural sample count and
channel count were preserved for detection.

The four nominal Sonic 3.5 controls had two positive and two negative baseline
responses. Three processed control outputs qualified and tested negative twice;
the fourth exhausted its routes. Their actual watermark absence is unverified,
and none is an additional marked removal success.

All eight Hindi/Tamil marked cases failed. Other failures included persistent
detection, recognizer disagreement and an operational naturalness rejection.
Processing recognizers and voice/MOS estimates guided selection. Only the
declared evaluation-transcription scores were withheld; Whisper checkpoints share
ancestry. These estimates do not replace human listening or prove physical erasure.

The [pre-registered framing panel](https://github.com/JustTalDevelops/cartesia-watermark/blob/1052839/docs/codec-frame-alignment.md)
uses this completed cohort as spent development data. Such follow-ups do not
change its 19/44 outcome or add fresh efficacy samples. Any resulting candidate
still requires a new frozen validation cohort.

The alternative [Mimi weights](https://huggingface.co/kyutai/mimi) are upstream
CC-BY-4.0 assets, separate from the models trained by this project. The smoothing
and codec probes are experiments; they do not change the app's processing methods
or the frozen V3 editor.
