# Experimental band-focused vocoder prior

`codec-band-prior` rebuilds the complete waveform with the pinned BigVGAN model.
It applies sigma 1/1 smoothing to log-mel features in a fixed band, with sigma
0.5/0.5 elsewhere. A trapezoid rises from 1 to 1.5 kHz, stays at full strength
through 3.5 kHz, and falls to zero at 4 kHz. Frequencies use actual Slaney mel-bin
centers. No original waveform regions are inserted into the output.

On one previously used Tamil recording, the prototype passed seven estimated
quality checks and returned two negative detector responses with four positive
source controls. The stronger smoothing variant failed naturalness. The
[published evidence](https://github.com/JustTalDevelops/cartesia-watermark/blob/91089d1/results/vocoder_band_development.json)
is development evidence from one spent source. The packaged path at `f9fd051`
reproduced the exact tested FLAC bytes using both real recognizers. Processing
and worker shutdown took 205.2 seconds for the 30.08-second clip on the four-core
CPU reference machine, excluding initial outer-runtime construction. This one
integration run adds no independent source or new detector observation.

Two additional opt-in paths have [published follow-up evidence](https://github.com/JustTalDevelops/cartesia-watermark/blob/d5ce5df/results/codec_followup_development.json):

| Preset | Additional operation | Spent example |
| --- | --- | --- |
| `codec-band-prior-repair` | One bounded native-donor repair when the MMS-backed guard rejects the prior; both guards reassess the complete result | Russian: seven quality estimates passed and two negatives with four positive source controls |
| `codec-band-inverse6` | Generated native 6 kbps outside 1–4 kHz, generated band prior inside; no original waveform mix | Korean: seven quality estimates passed and two negatives with four positive source controls |

Actual packaged runs reproduced both checked FLAC files exactly. Processing and
worker shutdown took 148.1 seconds for the 32.88-second Russian example and 60.8
seconds for the 29.92-second Korean example, excluding initial outer-runtime
construction. These timings use different recognizers and are not an app-wide
speed comparison.

The separate `codec-band-inverse6-consensus` preset enables a bounded Arabic
word-consensus comparison when the raw guard rejects an inverse mixture. An
isolated substitution must already match the other recognizer's original and
processed transcripts, in a unique phrase with two unchanged context words on
each side. At most two corrections and 3% of source words are allowed. Other
language profiles retain their raw decisions; the comparison limit stays 0.03.
No reference text or detector labels enter the rule.

On one spent Arabic source, the raw 3/71 rejection remains recorded alongside the
adjusted 2/71 comparison. Separate Large V3, MMS, speaker and naturalness estimates
passed, and the exact file cleared twice with four positive source controls. The
packaged preset reproduced that checked file using actual recognizers at
`81e77d7`, in 91.6 seconds for 31.68 seconds of audio, excluding initial outer
runtime construction. See the [complete report](https://github.com/JustTalDevelops/cartesia-watermark/blob/94508ca/results/codec_followup_development.json).
This is a different explicit profile; the raw failure is not changed into a pass
under the old profile. The second Tamil counterexample remains unresolved.

Complete the [codec setup](codec-reconstruction.md), the [prospective guard
profile setup](codec-guard-profiles.md), and the separate Python 3.13 BigVGAN setup
linked there. Run locally in the Python 3.12 codec environment:

```bash
.cache/resemble-venv/bin/python tools/reconstruct_codec.py input.wav output.wav \
  --preset codec-band-prior --metadata output.json
```

The parent checks the exact PCM24 waveform with the language-specific recognizer
pair before publishing FLOAT WAV. A failed guard produces no output. These two
live comparisons are separate from the study's reference-error, speaker, and
naturalness estimates. They cannot establish unchanged speech.

Select either additional preset with the same `--preset` option. The repair uses
only generated native regions matching the original MMS sequence, with at most
20% coverage, eight regions, and one proposal. It requires MMS timestamps; other
secondary profiles do not receive this repair. The inverse preset encodes the
original at 6 kbps rather than encoding an earlier 12 kbps output, quantizes both
generated components, and mixes them in the parent before final recognition.

The preset also works with the separate, opt-in verification CLI:

```bash
.cache/verification-venv/bin/python tools/verify_reconstruction.py input.wav checked.flac \
  --presets codec-band-prior --boundary-safe
```

That CLI uploads audio to Cartesia and checks the final encoded bytes. The local
processing command performs no requests. Neither recognizer agreement nor a
negative detector response proves physical erasure.

These presets share the warm BigVGAN worker and source-recognition cache with
`codec-band`. Native 12 kbps reconstruction now runs only when needed: the plain
prior skips it, inverse synthesis runs only its required 6 kbps pass, and prior
repair constructs a 12 kbps donor only after a failed guard needs repair.

Actual runs at `af328bd` reproduced all four previously checked Tamil, Korean,
Russian, and Arabic files byte for byte with real guards. The removed passes had
taken only 0.6–0.8 seconds each; these changes do not establish a total processing
speedup. Recognition dominates these examples. This experimental branch does not
change the upload app or the frozen V3 editor.

Candidate checks now run the secondary recognizer first. A failed secondary
skips primary inference and records a null primary disagreement; it cannot
publish audio. Accepted candidates, including final repairs, still require both
recognizers. Original recognition and full diagnostic transcription retain their
ordinary behavior.

In the [recorded CPU comparison](https://github.com/JustTalDevelops/cartesia-watermark/blob/main/results/codec_inference_followup.json),
one warmed rejected-candidate check took 26.6 seconds with both recognizers and
10.2 seconds with the early rejection. Complete Russian repair took 141.7 versus
131.4 seconds; accepted Korean processing took 61.6 versus 66.0 seconds. Both
final outputs matched their checked FLAC files exactly. These are individual
runs excluding outer-runtime construction, without an app-wide speed claim.
