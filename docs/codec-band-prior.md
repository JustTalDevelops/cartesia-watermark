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

The preset also works with the separate, opt-in verification CLI:

```bash
.cache/verification-venv/bin/python tools/verify_reconstruction.py input.wav checked.flac \
  --presets codec-band-prior --boundary-safe
```

That CLI uploads audio to Cartesia and checks the final encoded bytes. The local
processing command performs no requests. Neither recognizer agreement nor a
negative detector response proves physical erasure.

The prior shares the warm BigVGAN worker and source-recognition cache with
`codec-band`. The current outer runtime still constructs its native codec cache;
the prior does not mix that cached waveform into synthesis. This experimental
branch does not change the upload app or the frozen V3 editor.
