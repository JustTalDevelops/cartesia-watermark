"""All independent checks for a WaveNet detector checkpoint (no tuning on any of these):
  1. agreement with Cartesia on the 121 probe items (validate.py)
  2. agreement with Cartesia's distillation labels (transforms of new clips; never trained on)
  3. natural clips on held-out voices 24-29: per-model detection rate and AUC
"""
import glob, json, pathlib, sys
import numpy as np, soundfile as sf, torch
from sklearn.metrics import roc_auc_score
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
import train_detector as td, validate
from train_v3 import WaveNet

ROOT = pathlib.Path(__file__).resolve().parent.parent


@torch.no_grad()
def score(net, x, sr):
    s = td.segments(td.prep(np.asarray(x, np.float64), sr), hop=td.SEG // 4)
    return torch.sigmoid(net(torch.tensor(np.array(s)))).mean().item() if s else 0.0


def main(ckpt, thr=0.5):
    net = WaveNet(); net.load_state_dict(torch.load(ckpt)); net.eval()
    print(f"== 1. Cartesia probe set (121 items), checkpoint {ckpt}")
    validate.evaluate(net)
    print("== 2. Cartesia distillation labels (held out from training)")
    d = json.loads((ROOT / "results/distill.json").read_text())
    rows = []
    for k, v in d.items():
        x, sr = sf.read(ROOT / v["path"]); rows.append((v.get("family", "natural"), v["detected"], score(net, x, sr)))
    if rows:
        agree = np.mean([(p > thr) == y for _, y, p in rows])
        print(f"  agreement {agree:.3f} on {len(rows)} labels")
        for fam in sorted(set(r[0] for r in rows)):
            r = [x for x in rows if x[0] == fam]
            print(f"    {fam:10s} n={len(r):3d} agree={np.mean([(p > thr) == y for _, y, p in r]):.2f}")
    print("== 3. natural clips, held-out voices 24-29")
    sc = {}
    for p in sorted(glob.glob(str(ROOT / "audio/corpus3/*.wav"))):
        t, v, m = pathlib.Path(p).stem.split("_", 2)
        if int(v[1:]) >= 24:
            x, sr = sf.read(p); sc.setdefault(m, []).append(score(net, x, sr))
    for m, v in sc.items():
        print(f"  {m:14s} mean={np.mean(v):.2f}  flagged={np.mean(np.array(v) > thr):.2f}")
    y = [td.MARKED[m] for m in sc for _ in sc[m]]; s = [x for m in sc for x in sc[m]]
    print(f"  AUC {roc_auc_score(y, s):.3f}   accuracy@{thr} {np.mean((np.array(s) > thr) == np.array(y)):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
