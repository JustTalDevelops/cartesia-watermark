# Prospective codec study

The [input plan](../audio/codec_reliability_plan.json) declares sixteen new
generation requests: fourteen with marked model provenance and two matched
nominal controls, eleven utterance languages, fourteen project-unused public
voice IDs, a new long source and complete stereo derivatives. Voice primary
language mismatches are listed explicitly. This is a small cohort, not evidence
of universal removal. Source provenance and measured detector labels are separate.

The [freeze](../results/codec_reliability.freeze.json) binds the plan, processing,
evaluation code, model manifests and request ceilings before generation. Runtime
presets remain `codec-native` followed by `codec-band`. Both raw provider receipts
and complete canonical PCM24 FLAC sources are hashed. Canonicalization preserves
the whole sample grid; the two declared stereo controls use fixed channel gains.

## Completed result

The [complete report](../results/codec_reliability.json) retains all sixteen
requests. Every marked original was positive twice; both nominal controls were
negative twice. The candidate phase had twice-positive opening and closing
controls. All 66 HTTP attempts completed successfully.

| Marked sources | Result |
| --- | --- |
| 6/14 | Twice-negative final file and all seven quality criteria passed |
| 1/14 | Hindi cleared detection but Small reference error increased by 0.1184, exceeding 0.03 |
| 7/14 | No verified output: French, first Arabic, Korean, Russian, both Tamil sources, and the long source |

The two nominal controls passed processing and all quality criteria, but do not
count as removal transitions. Korean also had insufficient original-source
recognizer calibration under the frozen word-scoring policy. The long source
remained detected twice with each content-qualified preset. Six other sources
exhausted their presets because of detection or content-guard rejections.
The workflow did not meet the reliability target. Subsequent development probes
use these spent sources; they cannot change this study or add fresh evidence.

## Frozen procedure

Every original receives two whole-file detector checks and three original-only
recognizer evaluations before any candidate selection. Candidate checks begin and
end with twice-positive original controls. Each quality-qualified candidate gets
two responses on its exact final encoded bytes; positive counterexamples also get
a confirmation. Conflicting responses or API errors stop the phase for inspection.
The endpoint is globally paced at eighty seconds, with direct requests and saved
rate-limit cooldowns. All sixteen requests remain in the denominator.

Seven numerical output criteria are unchanged from development: Turbo paired
disagreement and reference-error increase at most 0.03, Small reference-error
increase at most 0.03, MMS paired disagreement and reference-error increase at
most 0.03, minimum speaker similarity across three ten-second windows at least
0.95, and DNSMOS overall change at least −0.2. All failures remain reported.
Turbo and MMS are processing guards; Small is an evaluation-only checkpoint.
These estimates do not provide human listening approval.

Reference error uses character units for Chinese/Japanese and whitespace word
units otherwise. Chinese references and hypotheses use pinned OpenCC 1.1.9
`t2s.json`; raw-script errors are retained too. Japanese is not Chinese-converted.
This prospective scoring correction follows original-only evidence of a
Traditional/Simplified mismatch. Runtime paired disagreement is unchanged.
Each checkpoint is calibrated separately against each original (error ≤0.15).
Source confidence requires at least two calibrated checkpoints. Insufficient
calibration remains unproven; it never removes a recording or cancels an output
criterion. Reference text, Small, speaker scores and MOS never select a fallback.

Install [evaluation dependencies](../requirements/codec_evaluation.txt) in the
Python 3.12 codec environment and [voice evaluation](../requirements/eval.txt) in
the separate Python 3.13 environment. The evaluation manifest pins additional
Whisper Small and WavLM files, which are upstream models rather than project
training outputs. Download the listed revisions into their declared local paths.
Processing and scoring load local weights only.

Generation requires an existing authorized `budget.json` ledger and lock directory;
the script refuses to initialize or reset one. It conservatively reserves twice
the UTF-8 byte count at $65 per million credits, enforces the shared $50 ceiling,
and never repeats a request with an existing reservation. A failed or uncertain
request remains fully reserved. The plan records the official pricing source and
the date checked. Generation is a paid operation.

From the repository root, the explicit stages are:

```bash
.venv/bin/python experiments/generate_codec_study.py \
  --plan audio/codec_reliability_plan.json \
  --freeze results/codec_reliability.freeze.json
.cache/verification-venv/bin/python experiments/run_codec_study.py \
  --plan audio/codec_reliability_plan.json \
  --freeze results/codec_reliability.freeze.json --phase materialize
```

Run `--phase baseline` in the lightweight environment. Separately run
`score_codec_study.py --originals-only` for `small`, `turbo` and `mms` in the
Python 3.12 environment, with the materialized `evaluation.plan.json`, a local
`--cache`, and outputs `scores/originals_<metric>.json`. Do not run two neural
jobs together on the four-core reference machine. The candidate phase requires
all of those completed receipts:

```bash
.cache/verification-venv/bin/python experiments/run_codec_study.py \
  --plan audio/codec_reliability_plan.json \
  --freeze results/codec_reliability.freeze.json --phase candidates
```

Finally score the selected files without `--originals-only` for all three
recognizers, and run `--metric voice` in Python 3.13. Missing output is a recorded
failure. Never use a failed final quality score to choose another preset within
this cohort. Raw transcripts and embeddings remain in ignored local caches;
public evidence contains scores, hashes and explicit failures. A completed phase
receipt cannot be silently overwritten or resumed by replaying API observations.
