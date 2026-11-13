"""Ensemble detector built from trained heads, configured by a JSON written by tools/calibrate_final.py.

A config lists its members, feature set and threshold. Current best: results/final_detector_r4.json
(FEAT=xlsr2; members x4c0-x4c2 = conv heads on XLS-R layer-2 frames trained with contrastive variants).
Members named "<tag>" load results/head_conv_heldout_<tag>.pt; "finetune_<round>" loads a fine-tuned
WavLM model (tools/finetune.py, round 2 only). Each member gives the mean logit over 3 s speech windows;
the clip score is the mean member logit.

    FEAT=xlsr2 python ...                               # FEAT must match the config (checked)
    from final_detector import Detector
    d = Detector(config="results/final_detector_r4.json"); score = d.score(audio, sr)   # > d.threshold => marked
"""
import json, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train_head as th
import frames as fr

ROOT = th.ROOT
CONFIG = ROOT / "results/final_detector.json"


class Detector:
    def __init__(self, members=None, config=CONFIG):
        config = pathlib.Path(config)
        cfg = json.loads(config.read_text()) if config.exists() else {}
        members = members or cfg["members"]
        self.threshold = cfg.get("threshold")
        assert cfg.get("feat", "wavlm1") == fr.FEAT or not cfg, f"config needs FEAT={cfg.get('feat')}"
        self.frozen, self.ft = [], None
        for m in members:
            if m.startswith("finetune"):                                # "finetune_r2" -> results/finetune_r2_best.pt
                import finetune
                net = finetune.Net(th.Conv().state_dict())
                net.load_state_dict(torch.load(ROOT / f"results/{m}_best.pt")); self.ft = net.eval()
            else:
                h = th.Conv(); h.load_state_dict(torch.load(ROOT / f"results/head_conv_heldout_{m}.pt")); self.frozen.append(h.eval())

    @torch.no_grad()
    def member_logits(self, x, sr):
        x = np.asarray(x, np.float64); x = x if x.ndim == 1 else x.mean(1)
        y16 = fr.to16k(x, sr)
        out = []
        if self.frozen:
            h, rms = fr.frames(y16)
            w = th.windows_of(h, rms)
            if not w:
                return None                                         # no speech above -50 dBFS
            xb = torch.tensor(np.array(w)).float()
            out += [float(net(xb).mean()) for net in self.frozen]
        if self.ft is not None:
            import finetune
            y = y16.astype(np.float32)
            ws = [finetune.znorm(finetune.window(y, s)) for s in finetune.speech_starts(y)]
            out.append(float(self.ft(torch.tensor(np.array(ws))).mean()))
        return out

    def score(self, x, sr):
        m = self.member_logits(x, sr)
        return None if m is None else float(np.mean(m))
