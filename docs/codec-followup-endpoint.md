# X-Codec2 and generated-donor repair endpoint panel

This is a development follow-up on all 25 marked failures of the completed 19/44 study. It retains 25 X-Codec2 outputs and both cached-repair donor slots for each source: 75 rows, including unrendered and processing-rejected treatments. It does not add fresh validation sources or change the completed study.

The completed X-Codec2 generation panel retained eight processing-eligible outputs and seventeen speech rejections. The separate cached-repair panel retained fifty proposal slots: sixteen needed no speech repair, eight had no compatible cached MMS alignment, one proposal was rejected, twenty rendered outputs failed speech checks, two failed voice estimates, and three passed processing checks. The three eligible repairs use generated Native donors on German 13, Hindi 18 and Tamil 43. Original waveform samples were not used as repair donors; reference text and detector labels did not guide proposals.

The [checker](../experiments/check_codec_followups.py) can declare this endpoint phase only after the preceding frame and TiCodec/band detector panels finish without errors. It validates their complete receipts and retains their exact-file history. It also validates every source, the full X-Codec2 generation evidence, the complete repair inventory, all rendered repair outputs, recognizer receipts, voice receipts and the original coverage limits.

The expected endpoint inventory contains eleven eligible unique files, each checked twice, plus two opening and two closing positive-source controls: **26 HTTP attempts maximum**. The declaration binds this count to the actual completed files. The shared minimum request-start interval is 80 seconds. Errors, contradictions, changed files or failed positive controls stop the phase without automatic retries. Identical candidate bytes are measured once per pair and remain linked to every applicable row.

```bash
python experiments/check_codec_followups.py declare \
  --generation .cache/removal-reliability/xcodec2_development/generation.json \
  --completed-study .cache/removal-reliability/codec_profiles_fresh/curated_complete_report.json \
  --declaration .cache/removal-reliability/codec_followup_endpoint/endpoint.declaration.json
python experiments/check_codec_followups.py detect \
  --declaration .cache/removal-reliability/codec_followup_endpoint/endpoint.declaration.json \
  --output .cache/removal-reliability/codec_followup_endpoint/detector.json
```

Independent reference scoring follows the complete endpoint phase and must include every eligible output, including those still detected. It uses the original authored reference plan and frozen `codec_profile_evaluation.py` checkpoint roles and `quality_criteria` arithmetic. Hindi uses Turbo for evaluation; Arabic uses Large and MMS; other languages use Small. These evaluation recognizers remain separate from each language's processing guards, although Whisper checkpoints share ancestry. The declaration also binds the reference plan, original reference scores and voice calibration.

Each guard's paired disagreement and each guard/evaluation reference-error increase must be at most 3%, or 1% for Hindi/Tamil. Speaker minimum remains 0.95 and overall DNSMOS delta at least -0.2. Source language must match the request, and at least two original recognizers must have calibrated reference error no greater than 15%, or 10% for Hindi/Tamil. Missing measurements cannot pass. All quality gates and two exact-file negative detector replies are required to qualify a spent transition.

The [scoring/export wrapper](../experiments/codec_followup_panel_report.py) is included in the declaration's code hashes. It copies original transcription-cache entries into a separate evaluation directory, checks every eligible output against the frozen references and retains all 75 rows when exporting results. Unrendered rows have no candidate checksum and cannot qualify. It refuses existing evaluation reports instead of overwriting them.

Eligibility and a negative detector reply alone do not qualify removal. The new continuous X-Codec2 experiment is a separate generation panel and is not part of this declaration. No new synthesis or paid compute is required by this follow-up.
