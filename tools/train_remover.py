"""Train an unpaired, detector-guided spectral editor on existing local audio.

    .venv/bin/python tools/train_remover.py --steps 400

No TTS generation or API queries. Voice-disjoint training/validation manifests
are saved alongside the checkpoint. sealed4 is excluded from both sets.
This optimizes detector suppression, not supervised watermark reconstruction.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from collections import Counter

os.environ["FEAT"] = "xlsr2"
import numpy as np
import soundfile as sf
import torch
torch.set_flush_denormal(True)
from torch import nn
from torch.nn import functional as F

import frames
import train_head as th
from remover import ROOT, Remover, SAMPLE_RATE, render_chunks16k

SEGMENT = 3 * SAMPLE_RATE


def stored_frame_values(h):
    """Match stored float16 features without float16 gradient underflow."""
    quantized = h.to(torch.float16).to(h.dtype)
    return h + (quantized-h).detach()


def suppression_loss(teacher, audio, marked):
    """Unmarked controls have no suppression term or detector backward pass."""
    result = audio.new_zeros(len(audio))
    if bool(marked.any()):
        selected = audio[marked]
        penalty = F.softplus(teacher(selected)+teacher.margin)
        if teacher.cross_encoder is not None:
            penalty = penalty + .5*F.softplus(teacher.cross_logits(selected)+teacher.margin)
        result = result.masked_scatter(marked, penalty)
    return result


def read_teacher_config(path, ancestors=()):
    """Carry every ensemble member's voice exclusions into student validation."""
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError('Cyclic teacher ensemble configuration')
    cfg = json.loads(path.read_text())
    if cfg.get('ensemble_configs'):
        children = [read_teacher_config(ROOT/member, (*ancestors, path))
                    for member in cfg['ensemble_configs']]
        for key in ('training_voices', 'calibration_voices'):
            cfg[key] = sorted(set(cfg.get(key, [])) | {v for child in children for v in child.get(key, [])})
    return cfg


class Teacher(nn.Module):
    """Frozen existing detector, with input gradients and identical frame padding."""
    def __init__(self, cross_detector=False, margin=2.0, spectral_weight=0.5, intelligibility_weight=0.0,
                 pesq_weight=0.0, pesq_target=4.4, config=None, waveform_weight=100., control_waveform_weight=500.):
        super().__init__()
        cfg = read_teacher_config(config or ROOT / "results/final_detector_r4.json")
        self.detector_config = cfg
        self.member_hashes = {}
        self.raw_oracle = None
        self.ensemble = nn.ModuleList()
        if cfg.get('ensemble_configs'):
            for index, member in enumerate(cfg['ensemble_configs']):
                child = Teacher(config=ROOT/member)
                self.ensemble.append(child)
                self.member_hashes[f'config_{index}'] = hashlib.sha256((ROOT/member).read_bytes()).hexdigest()
                self.member_hashes.update({f'{index}/{key}': value for key, value in child.member_hashes.items()})
            self.encoder = None
        elif cfg.get('raw_checkpoint'):
            from raw_oracle import load_raw_oracle
            raw_path = ROOT/cfg['raw_checkpoint']
            self.raw_oracle, _ = load_raw_oracle(raw_path)
            self.raw_oracle.eval().requires_grad_(False)
            self.member_hashes['raw_oracle'] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            self.encoder = None
        else:
            base_encoder = frames.model()
            self.encoder = (copy.deepcopy(base_encoder) if cfg.get('encoder_checkpoint') else base_encoder).eval().requires_grad_(False)
        self.heads = nn.ModuleList()
        if cfg.get('encoder_checkpoint'):
            oracle_path = ROOT / cfg['encoder_checkpoint']
            oracle = torch.load(oracle_path, map_location='cpu', weights_only=True)
            if oracle.get('format') != 'cartesia-encoder-oracle-v1':
                raise ValueError('Unsupported encoder oracle')
            self.encoder.load_state_dict(oracle['encoder_state_dict'])
            self.heads.append(th.Conv())
            self.heads.load_state_dict(oracle['heads_state_dict'])
            self.member_hashes['encoder_oracle'] = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
            self.heads.eval().requires_grad_(False)
        for member in ([] if cfg.get('encoder_checkpoint') or self.raw_oracle is not None or self.ensemble else cfg["members"]):
            head = th.Conv()
            member_path = ROOT / f"results/head_conv_heldout_{member}.pt"
            head.load_state_dict(torch.load(member_path,
                                            map_location="cpu", weights_only=True))
            self.member_hashes[member] = hashlib.sha256(member_path.read_bytes()).hexdigest()
            self.heads.append(head.eval().requires_grad_(False))
        self.threshold = cfg["threshold"]
        self.margin = margin
        self.waveform_weight = waveform_weight
        self.control_waveform_weight = control_waveform_weight
        self.spectral_weight = spectral_weight
        self.intelligibility_weight = intelligibility_weight
        self.intelligibility = None
        if intelligibility_weight:
            from third_party.stoi_loss import NegSTOILoss
            self.intelligibility = NegSTOILoss(16000, extended=True, do_resample=False).eval()
        self.pesq_weight, self.pesq_target, self.perceptual = pesq_weight, pesq_target, None
        if pesq_weight:
            from third_party.torch_pesq import PesqLoss
            self.perceptual = PesqLoss(1, sample_rate=16000).eval().requires_grad_(False)
        self.cross_encoder = None
        self.original_scores = {}
        self.cross_heads = nn.ModuleList()
        if cross_detector:
            from transformers import WavLMModel
            self.cross_encoder = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
            self.cross_encoder.encoder.layers = self.cross_encoder.encoder.layers[:1]
            self.cross_encoder.requires_grad_(False)
            cross_cfg = json.loads((ROOT / "results/final_detector_r3.json").read_text())
            self.cross_threshold = cross_cfg["threshold"]
            for member in cross_cfg["members"]:
                head = th.Conv()
                head.norm = nn.LayerNorm(768)
                head.net[0] = nn.Conv1d(768, 256, 5, padding=2)
                head.load_state_dict(torch.load(ROOT / f"results/head_conv_heldout_{member}.pt",
                                                map_location="cpu", weights_only=True))
                self.cross_heads.append(head.eval().requires_grad_(False))

    def forward(self, x):
        if self.ensemble:
            # Any positive member keeps this conservative training target
            # positive. A large negative from another member cannot cancel it.
            return torch.logsumexp(torch.stack([child(x)-child.threshold for child in self.ensemble]), dim=0)
        if self.raw_oracle is not None:
            return self.raw_oracle(x)
        z = (x - x.mean(-1, keepdim=True)) / (x.std(-1, correction=0, keepdim=True) + 1e-9)
        h = self.encoder(z, output_hidden_states=True).hidden_states[2]
        # frames.py stores features in float16, then the detector reads float32.
        return self.pool(h, x, self.heads, self.detector_config.get('temporal_pool', 'mean'))

    def cross_logits(self, x):
        z = (x - x.mean(-1, keepdim=True)) / (x.std(-1, correction=0, keepdim=True) + 1e-9)
        h = self.cross_encoder(z, output_hidden_states=True).hidden_states[1]
        return self.pool(h, x, self.cross_heads, self.detector_config.get('temporal_pool', 'mean'))

    @staticmethod
    def pool(h, x, heads, temporal_pool='mean'):
        h = stored_frame_values(h)
        starts = range(0, max(1, h.shape[1] - th.WIN + 1), th.HOP)
        windows, gates = [], []
        frames_count = h.shape[1]
        rms = x[:, :frames_count * 320].reshape(x.shape[0], frames_count, 320).square().mean(-1).sqrt()
        for start in starts:
            w = h[:, start:start + th.WIN]
            windows.append(F.pad(w, (0, 0, 0, max(0, th.WIN - w.shape[1]))))
            gates.append(rms[:, start:start + th.WIN].mean(-1) >= th.GATE)
        inputs = torch.cat(windows, 0)
        scores = torch.stack([head(inputs) for head in heads], -1).mean(-1).reshape(len(windows), x.shape[0]).T
        mask = torch.stack(gates, -1).to(scores.dtype)
        if temporal_pool == 'smoothmax':
            # Residual positive sections must remain visible even when many
            # already-negative sections lower the recording's mean score.
            temperature = .5
            active = mask.sum(-1).clamp_min(1)
            pooled = temperature * (torch.logsumexp((scores/temperature).masked_fill(mask == 0, -1e4), -1)
                                    - active.log())
            return torch.where(mask.any(-1), pooled, torch.zeros_like(pooled))
        if temporal_pool != 'mean':
            raise ValueError('Unsupported temporal pooling policy')
        return (scores * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def manifest(seed, train_limit, val_limit):
    plans = {c: json.loads((ROOT / f"audio/{c}_plan.json").read_text())
             for c in ("corpus5", "corpus6")}
    val_voices = th.C5_VAL_VOICES | th.C6_VAL_VOICES
    test_voices = {row[2] for row in json.loads((ROOT / "audio/corpus7_plan.json").read_text())
                   if row[0] == "sealed4"}
    groups = {"train": {True: [], False: []}, "validation": {True: [], False: []}}
    for corpus, plan in plans.items():
        for path in sorted((ROOT / "audio" / corpus).glob("*.wav")):
            model = path.stem.rsplit("_", 1)[-1]
            row = plan[int(path.stem.split("_")[1])]
            if row[0] != corpus or row[2] in test_voices:
                continue
            split = "validation" if row[2] in val_voices else "train"
            marked = model in th.MARKED
            groups[split][marked].append(dict(path=str(path.relative_to(ROOT)),
                voice=row[2], language=row[1], model=model, marked=marked,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rng = random.Random(seed)
    result = {"seed": seed, "purpose": "detector-guided suppression; no clean waveform targets"}
    for split, limit in (("train", train_limit), ("validation", val_limit)):
        selected = []
        for marked in (True, False):
            pool = groups[split][marked]
            rng.shuffle(pool)
            selected += pool[:round(limit * (0.75 if marked else 0.25))]
        rng.shuffle(selected)
        if not selected or not any(r["marked"] for r in selected):
            raise ValueError(f"No marked audio in {split}")
        result[split] = selected
    assert not ({r["voice"] for r in result["train"]} &
                {r["voice"] for r in result["validation"]})
    return result


def load_audio(rows, pad_to_segment=True):
    audio = []
    for row in rows:
        x, sr = sf.read(ROOT / row["path"])
        if x.ndim == 2:
            x = x.mean(1)
        y = frames.to16k(x, sr).astype(np.float32)
        if pad_to_segment and len(y) < SEGMENT:
            y = np.pad(y, (0, SEGMENT - len(y)))
        audio.append(y)
    return audio


def broad_manifest(seed, train_limit, val_limit, forbidden_val_voices=()):
    """Use every older corpus; reserve sealed4 and split all duplicates by voice ID."""
    plans = {n: json.loads((ROOT / f"audio/corpus{n}_plan.json").read_text()) for n in (5, 6, 7)}
    voices3 = json.loads((ROOT / "audio/corpus3/voices.json").read_text())
    voice_lists = {c: json.loads((ROOT / f"audio/{c}/voices.json").read_text())
                   for c in ("corpus4", "sealed")}
    forbidden_val_voices = set(forbidden_val_voices)
    val_voices = th.C5_VAL_VOICES | th.C6_VAL_VOICES | th.C7_VAL_VOICES
    val_voices |= {v for i, v in enumerate(voices3) if i >= 24 or i in (1, 5)}
    for corpus, vs in voice_lists.items():
        rendered = set()
        for path in (ROOT / "audio" / corpus).glob("*.wav"):
            parts = path.stem.split("_")
            language = re.match(r"[a-z]+", parts[0]).group()
            index = int(re.search(r"\d+$", parts[1]).group())
            rendered.add(vs[language][index])
        val_voices |= set(sorted(rendered - forbidden_val_voices)[::8])
    for n, name in ((5, "sealed2"), (6, "sealed3")):
        for language in {r[1] for r in plans[n] if r[0] == name}:
            val_voices |= set(sorted({r[2] for r in plans[n] if r[0] == name and r[1] == language} - forbidden_val_voices)[::8])
    val_voices -= forbidden_val_voices
    test_voices = {r[2] for r in plans[7] if r[0] == "sealed4"}
    groups = {"train": {True: [], False: []}, "validation": {True: [], False: []}}
    corpora = ("corpus3", "corpus4", "corpus5", "corpus6", "corpus7", "sealed", "sealed2", "sealed3")
    for corpus in corpora:
        for path in sorted((ROOT / "audio" / corpus).glob("*.wav")):
            parts = path.stem.split("_")
            model = parts[-1]
            if corpus == "corpus3":
                voice, language = voices3[int(parts[1][1:])], "en"
            elif corpus in voice_lists:
                language = re.match(r"[a-z]+", parts[0]).group()
                index = int(re.search(r"\d+$", parts[1]).group())
                voice = voice_lists[corpus][language][index]
            else:
                n = {"corpus5": 5, "corpus6": 6, "corpus7": 7, "sealed2": 5, "sealed3": 6}[corpus]
                row = plans[n][int(parts[1])]
                if row[0] != corpus:
                    raise ValueError(f"Plan mismatch: {path}")
                voice, language = row[2], row[1]
            if voice in test_voices:
                continue
            split = "validation" if voice in val_voices else "train"
            marked = model in th.MARKED
            groups[split][marked].append(dict(path=str(path.relative_to(ROOT)), voice=voice,
                language=language, model=model, marked=marked,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rng = random.Random(seed)
    result = {"seed": seed, "purpose": "multilingual balanced detector-guided suppression", "corpora": corpora,
              "initialization_train_voices": sorted(forbidden_val_voices)}
    for split, limit in (("train", train_limit), ("validation", val_limit)):
        selected = []
        for marked in (True, False):
            pools = {}
            for row in groups[split][marked]:
                pools.setdefault(row["language"], []).append(row)
            for rows in pools.values():
                rng.shuffle(rows)
            count = round(limit * (0.75 if marked else 0.25))
            while pools and count:
                for lang in sorted(list(pools)):
                    if count == 0:
                        break
                    selected.append(pools[lang].pop())
                    count -= 1
                    if not pools[lang]:
                        del pools[lang]
        rng.shuffle(selected)
        result[split] = selected
    assert not ({r["voice"] for r in result["train"]} & {r["voice"] for r in result["validation"]})
    return result


def crop(x, rng=None):
    if rng is None:
        start = (len(x) - SEGMENT) // 2
    else:
        candidates = rng.integers(0, len(x) - SEGMENT + 1, size=8)
        start = max(candidates, key=lambda s: np.mean(x[s:s + SEGMENT] ** 2))
    return x[start:start + SEGMENT]


def fidelity(x, y):
    power = x.square().mean(-1).clamp_min(1e-10)
    relative_mse = (y - x).square().mean(-1) / power
    spectral = torch.zeros_like(relative_mse)
    for size in (256, 1024):
        window = torch.hann_window(size, device=x.device)
        a = torch.stft(x, size, size // 4, window=window, return_complex=True).abs()
        b = torch.stft(y, size, size // 4, window=window, return_complex=True).abs()
        spectral = spectral + (a - b).abs().mean((1, 2)) / a.mean((1, 2)).clamp_min(1e-6)
    return relative_mse, spectral / 2


@torch.no_grad()
def validate(model, teacher, rows, audio, batch_size, whole_clips=False):
    model.eval()
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        x = torch.from_numpy(np.stack([a if whole_clips else crop(a) for a in audio[start:start + batch_size]]))
        y = render_chunks16k(model, x) if whole_clips else model(x)
        key = (tuple(row["path"] for row in batch), x.shape[-1])
        if key not in teacher.original_scores:
            before = teacher(x)
            cross_before = teacher.cross_logits(x) if teacher.cross_encoder is not None else None
            teacher.original_scores[key] = (before, cross_before)
        before, cross_before = teacher.original_scores[key]
        after = teacher(y)
        if teacher.cross_encoder is not None:
            cross_after = teacher.cross_logits(y)
        mse, spec = fidelity(x, y)
        if teacher.intelligibility is not None:
            proxy_estoi = -teacher.intelligibility(y, x)
        if teacher.perceptual is not None:
            proxy_pesq = teacher.perceptual.mos(x, y)
        for i, row in enumerate(batch):
            records.append({"path": row["path"], "marked": row["marked"],
                "before": float(before[i]), "after": float(after[i]),
                "snr_db": float(-10 * torch.log10(mse[i].clamp_min(1e-12))),
                "spectral_error": float(spec[i])})
            if teacher.cross_encoder is not None:
                records[-1].update(cross_before=float(cross_before[i]), cross_after=float(cross_after[i]))
            if teacher.intelligibility is not None:
                records[-1]["proxy_estoi"] = float(proxy_estoi[i])
            if teacher.perceptual is not None:
                records[-1]["proxy_pesq"] = float(proxy_pesq[i])
    marked = [r for r in records if r["marked"]]
    objective = np.mean([np.logaddexp(0, r["after"] + teacher.margin) +
                        (0.5 * np.logaddexp(0, r["cross_after"] + teacher.margin) if "cross_after" in r else 0) +
                        teacher.waveform_weight * 10 ** (-r["snr_db"] / 10) + teacher.spectral_weight * r["spectral_error"]
                        + teacher.intelligibility_weight * max(0, 0.995 - r.get("proxy_estoi", 1.0))
                        + teacher.pesq_weight * max(0, teacher.pesq_target - r.get("proxy_pesq", 5.0))
                        for r in marked])
    summary = {"objective": float(objective),
        "marked_before_rate": float(np.mean([r["before"] > teacher.threshold for r in marked])),
        "marked_after_rate": float(np.mean([r["after"] > teacher.threshold for r in marked])),
        "marked_mean_logit": float(np.mean([r["after"] for r in marked])),
        "marked_median_snr_db": float(np.median([r["snr_db"] for r in marked]))}
    if teacher.cross_encoder is not None:
        summary["cross_marked_after_rate"] = float(np.mean([r["cross_after"] > teacher.cross_threshold for r in marked]))
    if teacher.intelligibility is not None:
        summary["marked_median_proxy_estoi"] = float(np.median([r["proxy_estoi"] for r in marked]))
        summary["min_proxy_estoi"] = min(r["proxy_estoi"] for r in records)
    if teacher.perceptual is not None:
        summary["marked_median_proxy_pesq"] = float(np.median([r["proxy_pesq"] for r in marked]))
        summary["min_proxy_pesq"] = min(r["proxy_pesq"] for r in records)
    return summary, records


def selection_key(summary, pesq_floor=None, estoi_floor=None):
    """Fidelity feasibility precedes detector loss when quality floors are set.

    An infeasible model cannot win by improving classification. Normalized
    shortfalls also order infeasible models solely by speech preservation.
    These are training proxies; native reference metrics are evaluated separately.
    """
    shortfall = 0.0
    if pesq_floor is not None:
        if not np.isfinite(summary["min_proxy_pesq"]):
            return (True, float("inf"), float("inf"))
        shortfall += max(0, pesq_floor - summary["min_proxy_pesq"]) / 0.1
    if estoi_floor is not None:
        if not np.isfinite(summary["min_proxy_estoi"]):
            return (True, float("inf"), float("inf"))
        shortfall += max(0, estoi_floor - summary["min_proxy_estoi"]) / 0.01
    return (shortfall > 0, shortfall, summary["objective"])


def save_checkpoint(checkpoint, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    global SEGMENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-clips", type=int, default=192)
    parser.add_argument("--val-clips", type=int, default=24)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--architecture", choices=("spectral", "waveform"), default="spectral")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--resume", type=Path)
    parser.add_argument('--reuse-resume-validation', action='store_true',
                        help='Reuse the saved validation on resume when forward behavior and loss settings are unchanged')
    parser.add_argument("--init-checkpoint", type=Path, help="Start a new data split/run from existing weights")
    parser.add_argument("--broad", action="store_true", help="Use all older corpora with language-balanced sampling")
    parser.add_argument("--cross-detector", action="store_true", help="Also train against the WavLM ensemble")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--local-snr-db", type=float)
    parser.add_argument("--data-manifest", type=Path, help="Reuse an audited training/validation split")
    parser.add_argument("--context-layers", type=int)
    parser.add_argument("--spectral-weight", type=float, default=0.5)
    parser.add_argument("--intelligibility-weight", type=float, default=0.0)
    parser.add_argument("--pesq-weight", type=float, default=0.0)
    parser.add_argument("--pesq-target", type=float, default=4.4)
    parser.add_argument("--min-snr-db", type=float)
    parser.add_argument("--teacher-config", type=Path, default=ROOT / "results/final_detector_r4.json")
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument('--whole-clips', action='store_true', help='Train on complete native>=5s clips through inference chunking; batch size one')
    parser.add_argument("--waveform-gain", type=float)
    parser.add_argument("--waveform-weight", type=float, default=100.)
    parser.add_argument("--control-waveform-weight", type=float, default=500.)
    parser.add_argument("--max-phase", type=float)
    parser.add_argument("--max-gain", type=float)
    parser.add_argument("--residual-filter-half-length", type=int)
    parser.add_argument("--edit-low-hz", type=float)
    parser.add_argument("--edit-high-hz", type=float)
    parser.add_argument("--masking-margin-db", type=float)
    parser.add_argument("--extra-low-hz", type=float)
    parser.add_argument("--extra-high-hz", type=float)
    parser.add_argument("--extra-gain", type=float)
    parser.add_argument("--quality-pesq-floor", type=float, help="Minimum validation proxy PESQ before detector selection")
    parser.add_argument("--quality-estoi-floor", type=float, help="Minimum validation proxy ESTOI before detector selection")
    args = parser.parse_args()
    if args.reuse_resume_validation and not args.resume:
        parser.error('--reuse-resume-validation requires --resume')
    if args.whole_clips and args.batch_size != 1:
        parser.error('Complete-file training requires batch size one')
    if min(args.waveform_weight, args.control_waveform_weight) < 0:
        parser.error('Waveform preservation weights must be nonnegative')
    if args.max_phase is not None and not 0 <= args.max_phase <= np.pi:
        parser.error('Maximum phase must be between zero and pi radians')
    if args.max_gain is not None and not 0 <= args.max_gain <= 1:
        parser.error('Maximum gain must be between zero and one')
    if args.quality_pesq_floor is not None and not args.pesq_weight:
        parser.error("A quality PESQ floor requires --pesq-weight")
    if args.quality_estoi_floor is not None and not args.intelligibility_weight:
        parser.error("A quality ESTOI floor requires --intelligibility-weight")
    if not 1 <= args.segment_seconds <= 12:
        parser.error("Segment duration must be between 1 and 12 seconds")
    SEGMENT = int(args.segment_seconds * SAMPLE_RATE)
    if args.resume and args.init_checkpoint:
        parser.error("Choose --resume or --init-checkpoint")
    if min(args.steps, args.batch_size, args.validate_every, args.threads, args.width) < 1:
        parser.error("Steps, batch size, validation interval, threads and width must be positive")
    if args.train_clips < 4 or args.val_clips < 4 or args.lr <= 0:
        parser.error("Use at least four clips per split and a positive learning rate")
    if not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("Tag must be alphanumeric, optionally with underscores/hyphens")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = ROOT / "results" / f"remover_{args.tag}"
    out.parent.mkdir(exist_ok=True)
    data_path = out.with_suffix(".manifest.json")
    forbidden_val_voices = set()
    if args.init_checkpoint:
        init_manifest_path = args.init_checkpoint.with_suffix(".manifest.json")
        if not init_manifest_path.exists():
            parser.error("Initialization requires the original training manifest")
        init_manifest = json.loads(init_manifest_path.read_text())
        forbidden_val_voices = ({r["voice"] for r in init_manifest["train"]} |
                                set(init_manifest.get("initialization_train_voices", [])) |
                                set(init_manifest.get("oracle_calibration_voices", [])))
    elif args.resume and args.broad and data_path.exists():
        forbidden_val_voices = set(json.loads(data_path.read_text()).get("initialization_train_voices", []))
    teacher_metadata = read_teacher_config(args.teacher_config)
    teacher_calibration = (set(teacher_metadata.get("calibration_voices", [])) |
                           set(teacher_metadata.get("training_voices", [])))
    forbidden_val_voices |= teacher_calibration
    if args.data_manifest:
        data = json.loads(args.data_manifest.read_text())
        if {r["voice"] for r in data["validation"]} & ({r["voice"] for r in data["train"]} | forbidden_val_voices):
            parser.error("Supplied manifest leaks training/initialization voices into validation")
    else:
        data = (broad_manifest(args.seed, args.train_clips, args.val_clips, forbidden_val_voices)
                if args.broad else manifest(args.seed, args.train_clips, args.val_clips))
    data_json = json.dumps(data, indent=2)
    if data_path.exists() and data_path.read_text() != data_json:
        parser.error("Existing tag has a different manifest; use a new --tag")
    if out.with_suffix(".pt").exists() and not args.resume:
        parser.error("Checkpoint already exists; use --resume or a new --tag")
    data_path.write_text(data_json)
    train_audio, val_audio = load_audio(data["train"], not args.whole_clips), load_audio(data["validation"], not args.whole_clips)
    if args.whole_clips and any(len(x) < 5*SAMPLE_RATE for x in train_audio+val_audio):
        parser.error('Complete-file training requires native clips >=5 seconds; no end padding')
    initialize = args.resume or args.init_checkpoint
    checkpoint = torch.load(initialize, map_location="cpu", weights_only=True) if initialize else None
    if args.init_checkpoint:
        save_checkpoint(checkpoint, out.with_suffix(".init.pt"))
    model_config = dict(checkpoint["model_config"]) if checkpoint else dict(width=args.width, architecture=args.architecture)
    if args.context_layers is not None:
        model_config["context_layers"] = args.context_layers
    for key in ('edit_low_hz', 'edit_high_hz', 'masking_margin_db', 'extra_low_hz', 'extra_high_hz', 'extra_gain', 'max_phase', 'max_gain', 'residual_filter_half_length'):
        if getattr(args, key) is not None:
            model_config[key] = getattr(args, key)
    model = Remover(**model_config)
    if args.local_snr_db is not None:
        model.config["local_snr_db"] = args.local_snr_db
    if args.min_snr_db is not None:
        model.config["min_snr_db"] = args.min_snr_db
    if args.waveform_gain is not None:
        if not 0 < args.waveform_gain <= 2:
            parser.error('Waveform gain must be in (0, 2]')
        model.config['waveform_gain'] = args.waveform_gain
    if args.init_checkpoint:
        init_manifest_path = args.init_checkpoint.with_suffix(".manifest.json")
        if not init_manifest_path.exists():
            parser.error("Initialization requires the original training manifest")
        init_manifest = json.loads(init_manifest_path.read_text())
        if (forbidden_val_voices & {r["voice"] for r in data["validation"]}):
            parser.error("New validation voices overlap the initialization model's training voices")
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if incompatible.unexpected_keys or any(not k.startswith(("context.", "extra_head.")) for k in incompatible.missing_keys):
            parser.error("Initialization weights are incompatible with this model")
        if args.waveform_gain is not None and model.config['architecture'] == 'waveform':
            # Preserve the initial residual approximately when opening a larger
            # output range; avoid a sudden amplification of existing edits.
            ratio = checkpoint['model_config'].get('waveform_gain', .1)/args.waveform_gain
            with torch.no_grad():
                model.net[-1].weight.mul_(ratio)
                model.net[-1].bias.mul_(ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    initial_step, best = 0, (True, float("inf"), float("inf"))
    if args.resume:
        if checkpoint["manifest_sha256"] != hashlib.sha256(data_json.encode()).hexdigest():
            parser.error("Resume manifest differs from the saved run")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        initial_step = checkpoint["step"]
        best = selection_key(checkpoint["validation"], args.quality_pesq_floor, args.quality_estoi_floor)
        if out.with_suffix(".pt").exists():
            previous_best = torch.load(out.with_suffix(".pt"), map_location="cpu", weights_only=True)
            if previous_best["manifest_sha256"] != checkpoint["manifest_sha256"]:
                parser.error("Existing best checkpoint belongs to a different manifest")
            best = min(best, selection_key(previous_best["validation"], args.quality_pesq_floor, args.quality_estoi_floor))
        torch.set_rng_state(checkpoint["torch_rng"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        if args.steps <= initial_step:
            parser.error(f"Resume --steps must exceed the saved step ({initial_step})")
    teacher = Teacher(cross_detector=args.cross_detector, margin=args.margin,
                      spectral_weight=args.spectral_weight, intelligibility_weight=args.intelligibility_weight,
                      pesq_weight=args.pesq_weight, pesq_target=args.pesq_target, config=args.teacher_config,
                      waveform_weight=args.waveform_weight, control_waveform_weight=args.control_waveform_weight)
    if args.resume:
        # Frozen teacher construction initializes temporary layers before loading
        # their weights; restore the training RNG after those initializations.
        torch.set_rng_state(checkpoint['torch_rng'])
        rng.bit_generator.state = checkpoint['numpy_rng']
    probabilities = None
    if args.broad:
        counts = Counter((r["marked"], r["language"]) for r in data["train"])
        language_counts = Counter(marked for marked, lang in counts)
        probabilities = np.array([(0.75 if r["marked"] else 0.25) /
            language_counts[r["marked"]] / counts[r["marked"], r["language"]] for r in data["train"]])
        probabilities /= probabilities.sum()
    started = time.time()
    history_path = out.with_suffix(".history.json")
    history = json.loads(history_path.read_text()) if args.resume and history_path.exists() else []
    print(f"Loaded {len(train_audio)} train / {len(val_audio)} validation clips; "
          f"{sum(p.numel() for p in model.parameters())} trainable parameters", flush=True)
    if args.reuse_resume_validation:
        loss_keys = ('cross_detector', 'margin', 'spectral_weight', 'intelligibility_weight',
                     'pesq_weight', 'pesq_target', 'waveform_weight', 'control_waveform_weight',
                     'segment_seconds', 'whole_clips')
        previous = checkpoint.get('training_args', {})
        if (model.config != checkpoint['model_config'] or
                checkpoint.get('teacher_member_sha256') != teacher.member_hashes or
                any(previous.get(key) != getattr(args, key) for key in loss_keys)):
            parser.error('Saved validation cannot be reused with different model, teacher, or loss settings')
        summary = checkpoint['validation']
    else:
        summary, initial_records = validate(model, teacher, data["validation"], val_audio, args.batch_size, args.whole_clips)
    print(json.dumps({"step": initial_step, **summary}), flush=True)

    def snapshot(step, validation):
        return {"format": "cartesia-remover-v1", "state_dict": model.state_dict(),
                "model_config": model.config, "optimizer": optimizer.state_dict(),
                "step": step, "validation": validation, "seed": args.seed,
                "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                "manifest_sha256": hashlib.sha256(data_json.encode()).hexdigest(),
                "training_args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "teacher_config": teacher.detector_config, "teacher_member_sha256": teacher.member_hashes,
                'training_runtime': {'threads': torch.get_num_threads(), 'flush_denormal': True,
                                     'feature_quantization_gradient': 'float32_straight_through'},
                "limitations": "Unpaired detector suppression; no proof of watermark removal"}

    if initial_step == 0:
        # A quality-feasible starting model must beat any later infeasible
        # checkpoint, even when the latter has a lower detector objective.
        best = selection_key(summary, args.quality_pesq_floor, args.quality_estoi_floor)
        initial_checkpoint = snapshot(0, summary)
        save_checkpoint(initial_checkpoint, out.with_suffix('.pt'))
        save_checkpoint(initial_checkpoint, out.with_suffix('.last.pt'))
        out.with_suffix('.validation.json').write_text(json.dumps(initial_records, indent=2))
        print('Saved initial quality baseline', flush=True)
    running = []
    for step in range(initial_step + 1, args.steps + 1):
        model.train()
        indices = (rng.choice(len(train_audio), size=args.batch_size, p=probabilities)
                   if probabilities is not None else rng.integers(len(train_audio), size=args.batch_size))
        x = torch.from_numpy(np.stack([train_audio[i] if args.whole_clips else crop(train_audio[i], rng) for i in indices]))
        marked = torch.tensor([data["train"][i]["marked"] for i in indices])
        y = render_chunks16k(model, x) if args.whole_clips else model(x)
        mse, spectral = fidelity(x, y)
        detector_loss = suppression_loss(teacher, y, marked)
        preservation = torch.where(marked, teacher.waveform_weight, teacher.control_waveform_weight) * mse + teacher.spectral_weight * spectral
        loss = (detector_loss + preservation).mean()
        if teacher.intelligibility is not None:
            proxy_estoi = -teacher.intelligibility(y, x)
            if not torch.isfinite(proxy_estoi).all():
                raise RuntimeError('Non-finite training intelligibility estimate')
            if bool((proxy_estoi < .995).any()):
                loss = loss + teacher.intelligibility_weight * F.relu(0.995 - proxy_estoi).mean()
        if teacher.perceptual is not None:
            proxy_pesq = teacher.perceptual.mos(x, y)
            if not torch.isfinite(proxy_pesq).all():
                raise RuntimeError('Non-finite training perceptual estimate')
            if bool((proxy_pesq < teacher.pesq_target).any()):
                loss = loss + teacher.pesq_weight * F.relu(teacher.pesq_target - proxy_pesq).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        running.append(float(loss.detach()))
        if step % 10 == 0:
            print(f"step {step}/{args.steps} loss={np.mean(running[-10:]):.4f} "
                  f"elapsed={time.time()-started:.0f}s", flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            summary, records = validate(model, teacher, data["validation"], val_audio, args.batch_size, args.whole_clips)
            row = {"step": step, "seconds": time.time() - started,
                   "train_loss": float(np.mean(running)), **summary}
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2))
            print(json.dumps(row), flush=True)
            checkpoint = snapshot(step, summary)
            save_checkpoint(checkpoint, out.with_suffix(".last.pt"))
            candidate_key = selection_key(summary, args.quality_pesq_floor, args.quality_estoi_floor)
            if candidate_key < best:
                best = candidate_key
                save_checkpoint(checkpoint, out.with_suffix(".pt"))
                out.with_suffix(".validation.json").write_text(json.dumps(records, indent=2))
                print(f"Saved best checkpoint {out.with_suffix('.pt')}", flush=True)
            running = []


if __name__ == "__main__":
    main()
