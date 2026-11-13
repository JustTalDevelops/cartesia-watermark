"""Round 3 data: fresh sealed test set (sealed3) + more training voices (corpus6), one text per voice, 4 models.

    python tools/corpus6.py [char_cap] [--dry]

Voices come from the full /voices catalogue (each voice once, under its own language), excluding every voice
used in any earlier corpus or sealed set. sealed3 texts are new sentences that never appear in training.
Credit-safe: hard character cap, stops on the first 402.
"""
import json, random, sys, threading, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c, requests
import ast

ROOT = pathlib.Path(__file__).resolve().parent.parent
# training sentences: reuse corpus5's TRAIN_TEXTS (read as a literal; importing corpus5.py would start generation)
_src = ast.parse((ROOT / "tools/corpus5.py").read_text())
TRAIN_TEXTS = next(ast.literal_eval(n.value) for n in _src.body if isinstance(n, ast.Assign) and n.targets[0].id == "TRAIN_TEXTS")
MODELS = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
DRY = "--dry" in sys.argv
CAP = next((int(a) for a in sys.argv[1:] if a.isdigit()), 60000)
SEALED_TEXTS = {
    "en": ["The library will extend its opening hours during exam week, starting next Monday morning.",
           "We stopped at a small roadside stall and bought a bag of peaches for the long drive home.",
           "If the printer jams again, open the side panel and gently pull the paper towards you.",
           "My neighbour has been building a wooden boat in his garage for almost three years now."],
    "es": ["La farmacia de guardia de esta semana está en la plaza mayor, junto al ayuntamiento."],
    "fr": ["Le train pour Lyon partira avec vingt minutes de retard en raison d'un incident technique."],
    "hi": ["कल सुबह बारिश के कारण स्कूल की बस थोड़ी देर से आएगी, कृपया धैर्य रखें।"],
    "ja": ["明日の会議は午後二時からに変更になりましたので、資料の準備をお願いします。"],
    "de": ["Der Wochenmarkt findet wegen des Feiertags ausnahmsweise am Freitag statt."],
    "pt": ["O museu da cidade terá entrada gratuita no primeiro domingo de cada mês."],
    "ko": ["이번 주말에는 날씨가 맑아서 가족과 함께 공원으로 소풍을 갈 예정입니다."],
    "zh": ["由于道路施工，这条公交线路下周将临时改道，请乘客提前留意。"],
    "sv": ["Biblioteket har förlängt sina öppettider under sommaren, även på söndagar."],
    "he": ["הספרייה העירונית תהיה סגורה ביום שני בגלל עבודות שיפוץ בבניין."],
    "ar": ["سيتم إغلاق الطريق الرئيسي غداً صباحاً بسبب أعمال الصيانة، يرجى استخدام طريق بديل."],
}
SEALED_PLAN = {"en": 12, "es": 4, "fr": 3, "hi": 3, "ja": 2, "de": 2, "pt": 2, "ko": 2, "zh": 2, "sv": 2, "he": 3, "ar": 3}
TRAIN_PLAN = {"en": 100, "es": 25, "fr": 12, "hi": 12, "de": 6, "ja": 5, "pt": 4}

used = set(json.loads((ROOT / "audio/corpus3/voices.json").read_text()))
for vs in json.loads((ROOT / "audio/sealed/voices.json").read_text()).values():
    used |= set(vs)
used |= set(json.loads((ROOT / "audio/corpus4/voices.json").read_text())["en"])
used |= {r[2] for r in json.loads((ROOT / "audio/corpus5_plan.json").read_text())}

catalogue, after = {}, None
while True:
    p = {"limit": 100}
    if after:
        p["starting_after"] = after
    r = requests.get(f"{c.API}/voices", headers=c._headers(), params=p, timeout=30).json()
    for v in r["data"]:
        if v["id"] not in used:
            catalogue.setdefault(v.get("language"), []).append(v["id"])
    if not r.get("has_more"):
        break
    after = r["data"][-1]["id"]

rng = random.Random(2027)
plan = []                                        # (set, lang, voice, text)
for lang in sorted(set(SEALED_PLAN) | set(TRAIN_PLAN)):
    vs = sorted(catalogue.get(lang, [])); rng.shuffle(vs)
    ns = SEALED_PLAN.get(lang, 0); nt = TRAIN_PLAN.get(lang, 0)
    for i, v in enumerate(vs[:ns]):
        plan.append(("sealed3", lang, v, SEALED_TEXTS[lang][i % len(SEALED_TEXTS[lang])]))
    texts = TRAIN_TEXTS.get(lang, [])
    for i, v in enumerate(vs[ns:ns + nt]):
        plan.append(("corpus6", lang, v, texts[i % len(texts)]))
assert len({p[2] for p in plan}) == len(plan), "duplicate voice in plan"
plan.sort(key=lambda p: p[0] != "sealed3")       # sealed set first
if DRY:
    import collections
    print(collections.Counter((p[0], p[1]) for p in plan))
    print(f"{len(plan)} voices, ~{sum(len(t) for *_, t in plan) * len(MODELS)} characters"); raise SystemExit
for s in ("sealed3", "corpus6"):
    (ROOT / "audio" / s).mkdir(parents=True, exist_ok=True)
(ROOT / "audio/corpus6_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=0))
est = sum(len(t) for *_, t in plan) * len(MODELS)
print(f"{len(plan)} voices planned ({sum(p[0] == 'sealed3' for p in plan)} sealed), ~{est} characters", flush=True)

spent, stop, lock = [0], threading.Event(), threading.Lock()


def gen(job):
    s, lang, v, text, m, idx = job
    if stop.is_set():
        return
    p = ROOT / "audio" / s / f"{lang}_{idx:03d}_{m}.wav"
    if p.exists():
        return
    with lock:
        if spent[0] + len(text) > CAP:
            stop.set(); print("credit cap reached, stopping", flush=True); return
        spent[0] += len(text)
    try:
        c.tts(text, p, model=m, voice=v, language=lang)
    except Exception as e:
        if "402" in str(e) or "quota" in str(e):
            stop.set(); print("QUOTA EXCEEDED, stopping:", str(e)[:150], flush=True)
        else:
            print("FAILED", p.name, str(e)[:150], flush=True)


jobs = [(s, lang, v, t, m, i) for i, (s, lang, v, t) in enumerate(plan) for m in MODELS]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(gen, jobs))
print(f"done: ~{spent[0]} characters spent; sealed3={len(list((ROOT / 'audio/sealed3').glob('*.wav')))} "
      f"corpus6={len(list((ROOT / 'audio/corpus6').glob('*.wav')))} clips", flush=True)
