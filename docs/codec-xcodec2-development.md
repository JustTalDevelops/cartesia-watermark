# X-Codec2 development

This experiment tests a separately obtained [X-Codec2 model](https://huggingface.co/HKUSTAudio/xcodec2/tree/e412427ed30f0cf9d5e3c95562113deb10a32d03) on the 25 marked failures of the completed [19/44 study](../results/codec_profiles_reliability.json). These recordings are development data. The experiment does not change that study's results, and processing eligibility alone does not establish removal.

The weights belong to HKUSTAudio and carry CC-BY-NC-4.0 terms. They were not trained by this project and are not included in its model release. The upstream [author implementation](https://github.com/zhenye234/X-Codec-2.0) and Python package `xcodec2==0.1.3` provide the model architecture. The measured worker uses float32, four CPU threads, Transformers 4.46.3 and the pinned weights identified in the reports.

## Exact semantic-prefix execution

The author codec consumes semantic `hidden_states[16]`. The 24-layer semantic model normally continues through eight unused layers. [The optional context manager](../experiments/xcodec2_prefix.py) stops at that boundary and preserves the remaining seven LayerDrop RNG draws. It restores the original forward method and removes its hook when execution succeeds or raises an exception.

This helper is specific to the inspected evaluation-mode model, its hidden-state consumer and CPU configuration. It changes neither weights nor precision. The full codec must be compared with the optimized codec on the complete input before using it for evidence. The development driver requires exact code indices, decoded waveform samples and RNG advancement for every source channel. A mismatch stops the run.

The [synthetic preflight](../results/xcodec2_prefix_preflight.json) used the actual checkpoint and a five-second sine wave. Three paired repetitions produced identical code indices and waveform samples. A separate check matched RNG advancement. The ratio of median full time to median prefix time was 1.149, about 13% less codec time. This small CPU component benchmark excludes loading, recognition, APIs and application overhead; it does not measure speech quality or removal.

## Complete-panel export

The [completed generation report](../results/xcodec2_generation_development.json)
retains eight processing-eligible outputs and seventeen speech rejections from
all 25 sources. Eligible recordings were English 00, Spanish 04, French 10,
German 14, Japanese 20 and Chinese 32/33/35. Every Hindi/Tamil output failed
speech checks. No detector, independent reference or human-listening
qualification has been completed for these new outputs.

All 27 complete channel controls matched code indices, decoded samples and RNG
advancement exactly. Summed full/prefix codec times were 481.85 and 409.63 seconds,
respectively, about 15% less codec time. Full execution preceded prefix execution
in each pair; this was not a randomized timing study. Recognition, loading,
resampling, API pacing and app overhead are excluded. The result is a component
measurement, not an app-speed or removal claim.

The generation panel keeps all 25 sources, including speech and voice rejections. It preserves each recording's sample count, sample rate and channel count, processes complete channels independently, and uses the original speech and voice limits. No reference text guides generation or candidate selection. Detection and independent reference scoring are separate phases.

[The offline exporter](../experiments/xcodec2_generation_report.py) checks the frozen declaration, source and output checksums, sample grids, per-channel parity controls, recognizer receipts and operational voice receipts before producing a public report. It replays assessments with the recorded processing checkout, which must still be present unchanged; it rejects a different processing version already imported by the caller. It refuses partial runs and omits transcripts, embeddings, generated audio and local filesystem paths.

```bash
python experiments/xcodec2_generation_report.py \
  --directory /path/to/local/removal-reliability \
  --output /path/to/new-report.json
```

The private generation adapter is identified by its hash. This page and the prefix helper do not provide a standalone processing CLI or an installation workflow for the upstream model.

Run the lightweight evidence and hook-restoration tests in an environment with the existing codec research dependencies:

```bash
python -m unittest tools.test_xcodec2_reporting tools.test_xcodec2_prefix
```
