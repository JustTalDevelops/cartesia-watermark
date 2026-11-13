"""Sealed final test set: voices, texts and languages never used for training or validation.
Generate once; score the final detector on it once."""
import json, sys, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c, requests

OUT = pathlib.Path(__file__).resolve().parent.parent / "audio" / "sealed"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
TEXTS = {
    "en": ["The lighthouse keeper wrote in his journal every night, describing the ships, the weather, and the colour of the sea.",
           "Before submitting the form, double check that your address, phone number, and date of birth are all entered correctly.",
           "It is hard to believe that the little seedling we planted last spring has already grown taller than the garden fence."],
    "es": ["La biblioteca abre a las nueve de la mañana y cierra a las ocho de la noche, de lunes a sábado, durante todo el año.",
           "Mi abuela siempre decía que el secreto de una buena sopa es la paciencia y un poco de cariño en cada ingrediente."],
    "fr": ["Le train pour Lyon partira avec vingt minutes de retard en raison de travaux sur la voie entre Paris et Dijon.",
           "Chaque été, nous passions nos vacances dans une petite maison au bord de la mer, loin du bruit de la ville."],
}


def voices(lang, skip, n):
    r = requests.get(f"{c.API}/voices", headers=c._headers(), params={"limit": 100, "language": lang}, timeout=30).json()
    return [v["id"] for v in r["data"] if v["id"] not in skip][:n]


used = set(json.loads((OUT.parent / "corpus3" / "voices.json").read_text()))
V = {"en": voices("en", used, 12), "es": voices("es", used, 4), "fr": voices("fr", used, 4)}
(OUT / "voices.json").write_text(json.dumps(V))
jobs = [(lang, ti, vi, m) for lang in V for ti in range(len(TEXTS[lang])) for vi in range(len(V[lang])) for m in MODELS]


def gen(j):
    lang, ti, vi, m = j
    p = OUT / f"{lang}{ti}_{lang}v{vi:02d}_{m}.wav"
    if p.exists():
        return
    for _ in range(3):
        try:
            c.tts(TEXTS[lang][ti], p, model=m, voice=V[lang][vi], language=lang); return
        except Exception as e:
            err = e
    print("FAILED", p.name, str(err)[:200], flush=True)


with ThreadPoolExecutor(8) as ex:
    list(ex.map(gen, jobs))
print(len(list(OUT.glob("*.wav"))), "of", len(jobs), flush=True)
