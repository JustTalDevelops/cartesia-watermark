"""Fresh sealed test set (sealed2) + large many-voice training corpus (corpus5), one text per voice.

Credit-safe: stops at a hard character cap and on the first 402 (quota) response.
Voices are disjoint from every earlier set and from each other.
"""
import json, random, sys, threading, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c, requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 165000
TRAIN_TEXTS = {
    "en": ["The ferry leaves every hour from the north pier, weather permitting, and tickets can be bought on board.",
           "I honestly thought the meeting would be short, but we ended up talking about the budget for two hours.",
           "Water the tomatoes in the early morning so the leaves have time to dry before the afternoon sun.",
           "Your appointment has been moved to Thursday at half past three. Reply yes to confirm the new time.",
           "The old bridge was closed for repairs, so we had to drive twenty minutes around the lake instead.",
           "She kept every letter her grandfather wrote from the war, tied together with a faded blue ribbon."],
    "es": ["El autobús de las ocho siempre llega tarde, así que muchos prefieren caminar hasta la estación.",
           "Recuerde que la oficina estará cerrada el lunes por ser día festivo en toda la región.",
           "Mi hermano aprendió a tocar la guitarra solo, mirando videos por las noches después del trabajo."],
    "fr": ["La boulangerie du quartier ouvre à six heures, et il y a déjà du monde devant la porte.",
           "Merci de vérifier vos coordonnées avant de valider votre commande sur notre site.",
           "Nous avons marché le long de la rivière jusqu'au vieux moulin, puis nous sommes rentrés."],
    "de": ["Die Bibliothek bleibt wegen Renovierungsarbeiten bis Ende des Monats geschlossen.",
           "Gestern haben wir zum ersten Mal Schnee gesehen, und die Kinder waren völlig begeistert.",
           "Bitte melden Sie sich spätestens eine Woche vor dem Termin bei unserem Büro an."],
    "pt": ["A padaria da esquina abre às seis e o cheiro de pão quente toma conta da rua inteira.",
           "Lembre-se de levar um documento com foto no dia da prova, caso contrário não poderá entrar.",
           "Passamos a tarde inteira na praia, conversando e olhando os barcos passarem devagar."],
    "it": ["La farmacia di turno questa settimana è quella in piazza, aperta anche la domenica mattina.",
           "Abbiamo cenato in una piccola trattoria dove il proprietario cucinava ancora tutto da solo.",
           "Si ricorda ai passeggeri di non lasciare bagagli incustoditi all'interno della stazione."],
    "ja": ["駅前の新しいパン屋さんは毎朝七時に開店して、昼前にはほとんど売り切れてしまいます。",
           "明日の会議は午後二時からに変更になりましたので、資料の準備をお願いします。",
           "週末は家族で山に登って、頂上でお弁当を食べながらきれいな景色を楽しみました。"],
    "hi": ["कल सुबह से शहर में बारिश हो रही है, इसलिए कई स्कूल आज बंद रखे गए हैं।",
           "कृपया अपना टिकट और पहचान पत्र तैयार रखें, जांच के लिए इसकी ज़रूरत होगी।",
           "मेरी दादी हर रविवार को पूरे परिवार के लिए खास खाना बनाती थीं।"],
}
SEALED_TEXTS = {
    "en": ["The museum's night tours start at eight, and visitors are asked to bring their own small torch.",
           "We missed the last train home, so we shared a taxi with two strangers who turned out to be great company."],
    "es": ["Los vecinos organizaron una fiesta en la plaza para celebrar el final del verano."],
    "fr": ["Le marché du samedi est plus animé que jamais depuis l'arrivée des nouveaux producteurs."],
    "de": ["Am Sonntag findet im Stadtpark ein kleines Konzert unter freiem Himmel statt."],
    "ja": ["図書館で借りた本を返し忘れていたので、今日の帰りに必ず返却しようと思います。"],
    "hi": ["नई सड़क बनने के बाद गांव से शहर तक का सफ़र आधे घंटे में पूरा हो जाता है।"],
    "ko": ["오늘 저녁에는 친구들과 함께 강가를 따라 산책하면서 이야기를 나누기로 했어요."],
    "zh": ["周末我们去了郊外的果园，摘了很多新鲜的苹果，还在树下吃了午饭。"],
    "nl": ["De bakker op de hoek verkoopt elke ochtend vers brood en warme croissants."],
    "ru": ["Вчера вечером мы долго гуляли по набережной и смотрели, как зажигаются огни."],
    "pl": ["W sobotę na rynku odbędzie się festiwal muzyki ludowej i regionalnych potraw."],
    "sv": ["Vi tog båten ut till ön och åt lunch på en liten restaurang vid hamnen."],
    "tr": ["Hafta sonu sahilde uzun bir yürüyüş yaptık ve gün batımını izledik."],
}
SEALED_PLAN = {"en": 12, "es": 2, "fr": 2, "de": 2, "ja": 2, "hi": 2, "ko": 2, "zh": 1, "nl": 1, "ru": 1, "pl": 1, "sv": 1, "tr": 1}
TRAIN_PLAN = {"en": 170, "es": 40, "fr": 30, "de": 25, "pt": 20, "it": 15, "ja": 15, "hi": 15}

used = set(json.loads((ROOT / "audio/corpus3/voices.json").read_text()))
for vs in json.loads((ROOT / "audio/sealed/voices.json").read_text()).values():
    used |= set(vs)
used |= set(json.loads((ROOT / "audio/corpus4/voices.json").read_text())["en"][:26])


def catalogue(lang):
    out, after = [], None
    while True:
        p = {"limit": 100, "language": lang}
        if after:
            p["starting_after"] = after
        r = requests.get(f"{c.API}/voices", headers=c._headers(), params=p, timeout=30).json()
        out += [v["id"] for v in r["data"] if v["id"] not in used]
        if not r.get("has_more"):
            return out
        after = r["data"][-1]["id"]


rng = random.Random(2026)
plan = []                                        # (set, lang, voice, text)
for lang in sorted(set(SEALED_PLAN) | set(TRAIN_PLAN)):
    vs = catalogue(lang); rng.shuffle(vs)
    ns = SEALED_PLAN.get(lang, 0); nt = TRAIN_PLAN.get(lang, 0)
    for i, v in enumerate(vs[:ns]):
        plan.append(("sealed2", lang, v, SEALED_TEXTS[lang][i % len(SEALED_TEXTS[lang])]))
    for i, v in enumerate(vs[ns:ns + nt]):
        plan.append(("corpus5", lang, v, TRAIN_TEXTS[lang][i % len(TRAIN_TEXTS[lang])]))
plan.sort(key=lambda p: p[0] != "sealed2")       # sealed set first
for s in ("sealed2", "corpus5"):
    (ROOT / "audio" / s).mkdir(parents=True, exist_ok=True)
(ROOT / "audio/corpus5_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=0))
est = sum(len(t) for *_, t in plan) * len(MODELS)
print(f"{len(plan)} voices planned ({sum(p[0] == 'sealed2' for p in plan)} sealed), ~{est} characters", flush=True)

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
print(f"done: ~{spent[0]} characters spent; sealed2={len(list((ROOT / 'audio/sealed2').glob('*.wav')))} "
      f"corpus5={len(list((ROOT / 'audio/corpus5').glob('*.wav')))} clips", flush=True)
