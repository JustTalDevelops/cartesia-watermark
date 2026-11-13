"""Larger corpus for the local surrogate: 2 watermarked vs 2 unwatermarked models."""
import sys, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
import cartesia as c

TEXTS = [
    "The train to Edinburgh was delayed by forty minutes, so we bought coffee and watched the rain streak across the platform windows.",
    "Please remember to bring your identification card, a printed copy of your booking, and any medication you may need during the trip.",
    "Scientists believe the comet last passed close to the Earth more than three thousand years ago, long before written records began.",
    "Our customer support team is available Monday through Friday from eight in the morning until six in the evening, Eastern time.",
    "She opened the old wooden box slowly, and inside, wrapped in faded silk, was a silver compass that still pointed faithfully north.",
    "Interest rates are expected to remain unchanged this quarter, although several analysts warn that inflation could rise again next year.",
    "Preheat the oven to two hundred degrees, mix the flour with the butter until crumbly, then add the sugar and a pinch of salt.",
    "The museum's new exhibition explores how ancient sailors navigated the open ocean using only the stars, the wind, and the waves.",
    "If you hear the alarm, leave the building immediately by the nearest exit and gather at the assembly point in the north car park.",
    "Honestly, I did not expect the ending at all. I thought the detective had it figured out, but the twist in the last chapter got me.",
    "The software update improves battery life, fixes several bugs in the calendar app, and adds support for three new languages.",
    "Across the valley, the bells of the small church rang out, and one by one the lights in the village houses began to flicker on.",
]
VOICES = {"daniel": "47c38ca4-5f35-497b-b1a3-415245fb35e1", "jacqueline": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
          "clive": "b24f41fd-00a3-4cd8-992a-a0c9f13f3ef1", "katie": "f786b574-daa5-4673-aa0c-cbe3e8534c02"}
MODELS = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
OUT = pathlib.Path(__file__).resolve().parent.parent / "audio" / "corpus2"
OUT.mkdir(parents=True, exist_ok=True)

jobs = [(ti, vn, m) for ti in range(len(TEXTS)) for vn in VOICES for m in MODELS]


def gen(j):
    ti, vn, m = j
    p = OUT / f"t{ti:02d}_{vn}_{m}.wav"
    if p.exists():
        return
    for _ in range(3):
        try:
            c.tts(TEXTS[ti], p, model=m, voice=VOICES[vn]); return
        except Exception as e:
            err = e
    print("FAILED", p.name, err, flush=True)


with ThreadPoolExecutor(4) as ex:
    list(ex.map(gen, jobs))
print(len(list(OUT.glob("*.wav"))), "of", len(jobs))
