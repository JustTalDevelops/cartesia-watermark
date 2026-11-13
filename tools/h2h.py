"""Head-to-head on unpadded audio: sealed4 voices, two clips joined into one ~8 s clip.

    python tools/h2h.py build                       # audio/h2h4/<lang>_<idx>_{marked,clean}.wav
    python tools/h2h.py score <config.json>         # ours vs Cartesia (results/h2h4_cartesia.json), provenance truth

marked = sonic-3.6 clip + 0.3 s gap + sonic-preview clip of the same voice; clean = sonic-3.5 + gap + sonic-3.
Cartesia's detector needs >= 5 s, so short single clips had to be padded with silence for it (which hurts it);
these joined clips need no padding. No model choice depends on this set: the detector config is fixed already.
"""
import json, pathlib, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "audio/sealed4", ROOT / "audio/h2h4"


def build():
    OUT.mkdir(exist_ok=True)
    stems = sorted({p.name.rsplit("_", 1)[0] for p in SRC.glob("*.wav")})
    for st in stems:
        for name, (a, b) in (("marked", ("sonic-3.6", "sonic-preview")), ("clean", ("sonic-3.5", "sonic-3"))):
            xa, sr = sf.read(SRC / f"{st}_{a}.wav"); xb, sr2 = sf.read(SRC / f"{st}_{b}.wav"); assert sr == sr2
            sf.write(OUT / f"{st}_{name}.wav", np.concatenate([xa, np.zeros(int(0.3 * sr)), xb]).astype(np.float32), sr, subtype="FLOAT")
    print(len(list(OUT.glob("*.wav"))), "clips")


def score(config):
    from sklearn.metrics import roc_auc_score
    from final_detector import Detector
    det = Detector(config=ROOT / config)
    cart = {k.split("/")[-1]: v for k, v in json.loads((ROOT / "results/h2h4_cartesia.json").read_text()).items()}
    rows = []
    for p in sorted(OUT.glob("*.wav")):
        x, sr = sf.read(p); s = det.score(x, sr)
        rows.append(dict(clip=p.name, marked=p.stem.endswith("marked"), score=s, ours=s > det.threshold,
                         cartesia=cart.get(p.name), seconds=len(x) / sr))
    (ROOT / "results/h2h4_final.json").write_text(json.dumps(rows, indent=0))
    y = np.array([r["marked"] for r in rows]); o = np.array([r["ours"] for r in rows]); c = np.array([bool(r["cartesia"]) for r in rows])
    s = np.array([r["score"] for r in rows])
    print(f"{len(rows)} joined clips ({y.sum()} marked), median {np.median([r['seconds'] for r in rows]):.1f} s, "
          f"Cartesia labels for {sum(r['cartesia'] is not None for r in rows)}")
    print(f"  ours     : AUC {roc_auc_score(y, s):.3f}  TPR {o[y].mean():.3f}  FPR {o[~y].mean():.3f}  accuracy {(o == y).mean():.3f}")
    print(f"  Cartesia : TPR {c[y].mean():.3f}  FPR {c[~y].mean():.3f}  accuracy {(c == y).mean():.3f}")
    print(f"  agreement: {(o == c).mean():.3f}")
    for r in rows:
        if r["ours"] != r["marked"] or bool(r["cartesia"]) != r["marked"]:
            print(f"    {r['clip']:24s} truth={'M' if r['marked'] else 'c'} ours={r['score']:+.2f} cartesia={r['cartesia']}")


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else score(sys.argv[2])
