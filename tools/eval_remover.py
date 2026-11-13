"""Evaluate a fixed remover on complete held-out files, with before/after scores.

    .venv/bin/python tools/eval_remover.py --dataset sealed4
    .venv/bin/python tools/eval_remover.py --dataset h2h4

Writes edited FLOAT WAVs and a JSON report. No API queries. The XLS-R detector
was used for training; its score measures suppression, not independent proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["FEAT"] = "xlsr2"
import numpy as np
import soundfile as sf
import torch
from pystoi import stoi

from final_detector import Detector
from remover import ROOT, DEFAULT_CHECKPOINT, load_model, process_audio, quality_metrics, resample, native_spectral_metrics
try:
    from pesq import pesq, PesqError
except ImportError:
    pesq = None


def speech_metrics(original, edited, sample_rate):
    ref = original if original.ndim == 1 else original.mean(1)
    out = edited if edited.ndim == 1 else edited.mean(1)
    metrics = {"stoi": float(stoi(ref, out, sample_rate, extended=False)),
               "estoi": float(stoi(ref, out, sample_rate, extended=True)), "pesq_wb": None}
    if pesq is not None:
        score = float(pesq(16000, resample(ref, sample_rate, 16000),
                           resample(out, sample_rate, 16000), "wb", on_error=PesqError.RETURN_VALUES))
        if score > 0:
            metrics["pesq_wb"] = score
        else:
            metrics["pesq_error"] = score
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", choices=("sealed4", "h2h4", "validation"), default="sealed4")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument('--quality-only', action='store_true', help='Measure fidelity without running the local detector')
    parser.add_argument("--plan",type=Path,help="Frozen subset plan for a sealed test batch")
    args = parser.parse_args()
    if not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("Use an alphanumeric tag")
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.checkpoint)
    detector = None if args.quality_only else Detector(config=ROOT / "results/final_detector_r4.json")
    source = ROOT / "audio" / ("remover_validation" if args.dataset == "validation" else args.dataset)
    paths = sorted(source.glob("*.wav"))
    if not paths:
        parser.error("Dataset contains no WAV files")
    manifest_path = args.checkpoint.with_suffix(".manifest.json")
    if not manifest_path.exists():
        parser.error("Checkpoint manifest is required to audit voice overlap")
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != checkpoint["manifest_sha256"]:
        parser.error("Manifest hash differs from checkpoint")
    excluded = {r["voice"] for split in ("train", "validation") for r in manifest[split]}
    inherited = (set(manifest.get("initialization_train_voices", [])) |
                 set(manifest.get("oracle_calibration_voices", [])) |
                 set(checkpoint.get("teacher_config", {}).get("calibration_voices", [])) |
                 set(checkpoint.get("teacher_config", {}).get("training_voices", [])))
    excluded |= inherited
    plan = json.loads((ROOT / "audio/corpus7_plan.json").read_text())
    test_voices = {r[2] for r in plan if r[0] == "sealed4"}
    if args.plan:
        if args.dataset=='validation':parser.error('Validation uses its original fixed plan')
        subset=json.loads(args.plan.read_text())
        if subset['dataset']!=args.dataset:parser.error('Test plan dataset mismatch')
        paths=[];test_voices=set()
        for row in subset['items']:
            path=ROOT/row['path']
            if path.resolve().parent!=source.resolve():parser.error('Test plan contains a different dataset')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:parser.error('Test input changed since selection')
            entry=plan[int(path.stem.split('_')[1])]
            if entry[0]!='sealed4' or entry[2]!=row['voice']:parser.error('Test voice plan mismatch')
            paths.append(path);test_voices.add(row['voice'])
    if args.dataset == "validation":
        excluded = {r["voice"] for r in manifest["train"]} | set(manifest.get("initialization_train_voices", []))
        excluded |= inherited
        validation_plan = json.loads((source / "plan.json").read_text())
        paths = [ROOT / row['path'] for row in validation_plan['items']]
        test_voices = {r["voice"] for r in validation_plan["items"]}
        for row in validation_plan["items"]:
            if hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                parser.error("Validation audio differs from its fixed plan")
    if test_voices & excluded:
        parser.error("Evaluation voices overlap training/validation")
    output_dir = ROOT / "audio" / f"remover_{args.tag}" / args.dataset
    report_path = ROOT / "results" / f"remover_{args.tag}_{args.dataset}.json"
    if report_path.exists():
        parser.error("Report already exists; use a new --tag for an explicit new evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, started = [], time.time()
    for i, path in enumerate(paths):
        x, sr = sf.read(path)
        marked = path.stem.endswith(("sonic-3.6", "sonic-preview", "_marked"))
        y = process_audio(model, x, sr)
        out = output_dir / path.name
        # Score the encoded file, which is exactly what the CLI returns.
        sf.write(out, y, sr, subtype="FLOAT")
        y, _ = sf.read(out)
        before, after = (detector.score(x, sr), detector.score(y, sr)) if detector is not None else (None, None)
        intelligibility = speech_metrics(x, y, sr)
        rows.append({"source": str(path.relative_to(ROOT)), "output": str(out.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "marked": marked, "seconds": len(x) / sr, "sample_rate": sr,
            "before": before, "after": after, **quality_metrics(x, y), **intelligibility,
            **native_spectral_metrics(x, y, sr)})
        if getattr(model, 'last_quality_decision', None) is not None:
            rows[-1]['quality_decision'] = dict(model.last_quality_decision)
        if (i + 1) % 16 == 0:
            print(f"{i+1}/{len(paths)} clips; elapsed {time.time()-started:.0f}s", flush=True)
    marked = [r for r in rows if r["marked"]]
    control = [r for r in rows if not r["marked"]]
    scored_marked=[r for r in marked if r['before'] is not None and r['after'] is not None]
    scored_control=[r for r in control if r['before'] is not None and r['after'] is not None]
    detected = [r for r in scored_marked if r["before"] > detector.threshold]
    pesq_scores = [r["pesq_wb"] for r in marked if r["pesq_wb"] is not None]
    summary = {"marked_clips": len(marked), "control_clips": len(control),
        "marked_scored_clips":len(scored_marked),
        "marked_detected_before": sum(r["before"] > detector.threshold for r in scored_marked),
        "marked_detected_after": sum(r["after"] > detector.threshold for r in scored_marked),
        "previously_detected_now_negative": sum(r["after"] <= detector.threshold for r in detected),
        "marked_median_snr_db": float(np.median([r["snr_db"] for r in marked])),
        "marked_min_snr_db": float(min(r["snr_db"] for r in marked)),
        "marked_median_stoi": float(np.median([r["stoi"] for r in marked])),
        "marked_min_stoi": float(min(r["stoi"] for r in marked)),
        "marked_median_estoi": float(np.median([r["estoi"] for r in marked])),
        "marked_pesq_clips": len(pesq_scores),
        "marked_median_pesq_wb": float(np.median(pesq_scores)) if pesq_scores else None,
        "marked_min_pesq_wb": float(min(pesq_scores)) if pesq_scores else None,
        "marked_median_native_spectral_change_db": float(np.median([r['native_active_bin_rms_change_db'] for r in marked])),
        "marked_max_native_band_gain_change_db": float(max(r['native_max_active_band_gain_change_db'] for r in marked)),
        "control_median_snr_db": float(np.median([r["snr_db"] for r in control])) if control else None,
        "control_detected_before": sum(r["before"] > detector.threshold for r in scored_control),
        "control_detected_after": sum(r["after"] > detector.threshold for r in scored_control),
        "control_median_stoi":float(np.median([r['stoi'] for r in control])) if control else None,
        "control_median_pesq_wb":float(np.median([r['pesq_wb'] for r in control if r['pesq_wb'] is not None])) if any(r['pesq_wb'] is not None for r in control) else None}
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint["step"], "dataset": args.dataset,
        "used_for_model_selection": args.dataset == "validation",
        "voice_overlap": 0, "threshold": detector.threshold if detector is not None else None,
        'local_detector_evaluated': detector is not None,
        "limitation": "Training detector; negative scores do not prove watermark removal",
        "summary": summary, "clips": rows}
    if detector is None:
        for key in ('marked_detected_before', 'marked_detected_after', 'previously_detected_now_negative',
                    'control_detected_before', 'control_detected_after'):
            summary[key] = None
    if args.plan:report['evaluation_plan_sha256']=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
