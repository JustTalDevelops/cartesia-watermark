"""Many-voice corpus: 30 voices x 6 texts x {2 marked, 2 unmarked} models, for the local detector."""
import json, sys, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c, requests

TEXTS = [
    "We walked along the river until the path disappeared into the reeds, and then we simply sat and listened to the water.",
    "Your order has been shipped and should arrive within three to five business days. You can track it from your account page.",
    "The committee will review every application carefully before announcing the final decision at the end of next month.",
    "When the storm finally passed, the whole town came outside to look at the fallen trees and the strange, bright orange sky.",
    "Add two cups of water, bring it to a boil, then lower the heat and let it simmer gently for about twenty minutes.",
    "Nobody in the office could explain why the printer only worked on Tuesdays, but by now everyone had simply accepted it.",
]
MODELS = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
OUT = pathlib.Path(__file__).resolve().parent.parent / "audio" / "corpus3"
OUT.mkdir(parents=True, exist_ok=True)
r = requests.get(f"{c.API}/voices", headers=c._headers(), params={"limit": 100, "language": "en"}, timeout=30).json()
voices = [v["id"] for v in r["data"]][:30]
(OUT / "voices.json").write_text(json.dumps(voices))
jobs = [(ti, vi, m) for ti in range(len(TEXTS)) for vi in range(len(voices)) for m in MODELS]


def gen(j):
    ti, vi, m = j
    p = OUT / f"t{ti}_v{vi:02d}_{m}.wav"
    if p.exists():
        return
    for _ in range(3):
        try:
            c.tts(TEXTS[ti], p, model=m, voice=voices[vi]); return
        except Exception as e:
            err = e
    print("FAILED", p.name, err, flush=True)


with ThreadPoolExecutor(8) as ex:
    list(ex.map(gen, jobs))
print(len(list(OUT.glob("*.wav"))), "of", len(jobs), flush=True)
