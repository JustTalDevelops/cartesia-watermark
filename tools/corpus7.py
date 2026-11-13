"""Round 4 data: fresh sealed test set (sealed4, mostly non-English, Tamil + Tagalog never used for training)
plus a multilingual training corpus (corpus7) with more sentence variety. One text per voice, 4 models.

    python tools/corpus7.py [char_cap] [--dry]

Voices come from the full /voices catalogue (each once, under its own language), excluding every voice used in
any earlier corpus or sealed set. sealed4 sentences never appear in training. Training jobs are shuffled by voice,
so if the character cap cuts the run short no language is dropped wholesale. Stops on the first 402.
"""
import json, random, sys, threading, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c, requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
DRY = "--dry" in sys.argv
CAP = next((int(a) for a in sys.argv[1:] if a.isdigit()), 72000)

SEALED_TEXTS = {
    "en": ["The pharmacy closes early on Sundays, so pick up your order before noon.",
           "We missed the last bus and had to walk home along the river in the dark.",
           "Your parcel was left with the neighbour at number twelve this afternoon.",
           "The choir rehearses in the old church hall every Wednesday evening."],
    "es": ["El mercado del barrio abre los sábados desde las siete de la mañana."],
    "fr": ["La piscine municipale sera fermée toute la semaine pour nettoyage."],
    "hi": ["आज शाम को बाज़ार में बहुत भीड़ थी, इसलिए हम जल्दी घर लौट आए।"],
    "de": ["Unser Zug hatte wieder Verspätung, also haben wir einen Kaffee getrunken."],
    "ja": ["駅前の新しいパン屋さんは、朝七時から営業しています。"],
    "pt": ["A biblioteca municipal vai abrir mais cedo durante as férias escolares."],
    "ar": ["افتتح المتحف الجديد أبوابه للزوار في نهاية الأسبوع الماضي."],
    "he": ["הרכבת לירושלים יוצאת כל חצי שעה מהרציף השני."],
    "ko": ["새로 생긴 빵집은 아침 일곱 시부터 문을 엽니다."],
    "zh": ["图书馆周末开放时间延长到晚上九点。"],
    "ru": ["Новая библиотека в нашем районе открывается в следующий понедельник."],
    "ta": ["நாளை காலை பள்ளிக்கு விடுமுறை என்று அறிவிக்கப்பட்டுள்ளது.",
           "எங்கள் ஊரில் புதிய நூலகம் அடுத்த வாரம் திறக்கப்படும்."],
    "tl": ["Magsasara nang maaga ang palengke bukas dahil sa pista.",
           "Sumakay kami ng bangka papunta sa maliit na isla kahapon."],
}
TRAIN_TEXTS = {
    "en": ["Please remember to switch off the lights when you leave the office tonight.",
           "The museum is free on the first Sunday of every month, so it gets crowded.",
           "I found an old photograph of my parents tucked inside a library book.",
           "Our flight was delayed by two hours because of fog over the airport.",
           "The recipe calls for three eggs, but I only had two, so I improvised.",
           "Thanks for calling. Your estimated wait time is less than five minutes.",
           "He spent the whole summer repairing the fence around his grandmother's farm.",
           "If it rains tomorrow, the football match will be moved to the indoor hall."],
    "es": ["Mañana por la mañana habrá cortes de agua en el centro de la ciudad.",
           "Mi abuela siempre prepara una tortilla enorme cuando venimos de visita.",
           "Por favor, deje su mensaje después de la señal y le llamaremos pronto.",
           "El tren hacia Valencia sale del andén número cuatro en diez minutos."],
    "fr": ["Le marché de Noël ouvrira ses portes la semaine prochaine sur la grande place.",
           "Nous avons oublié les clés dans la voiture, alors nous avons appelé un voisin.",
           "Votre rendez-vous chez le dentiste est confirmé pour jeudi à quinze heures.",
           "Il a plu toute la nuit, et le jardin est maintenant couvert de feuilles."],
    "hi": ["कल सुबह दस बजे बैंक में एक ज़रूरी बैठक होगी।",
           "मेरी बहन ने इस साल पहली बार अपने बगीचे में टमाटर उगाए।",
           "कृपया अपना टिकट और पहचान पत्र तैयार रखें।",
           "बारिश के कारण आज की क्रिकेट प्रतियोगिता रद्द कर दी गई है।"],
    "ar": ["سيبدأ العرض المسرحي في تمام الساعة الثامنة مساءً.",
           "نسيت مظلتي في المقهى، ثم بدأ المطر فجأة.",
           "يرجى الحضور قبل موعدكم بخمس عشرة دقيقة.",
           "زرنا السوق القديم واشترينا بعض التوابل والحلويات."],
    "he": ["ההרצאה תתחיל בשעה שבע בערב באולם הגדול.",
           "שכחתי את המטרייה בבית הקפה, ואז התחיל לרדת גשם.",
           "נא להגיע עשר דקות לפני מועד התור.",
           "ביום שישי הלכנו לשוק וקנינו פירות טריים."],
    "ja": ["明日は雨の予報なので、傘を忘れないでください。",
           "週末に家族と一緒に海へ出かけました。",
           "お電話ありがとうございます。担当者におつなぎします。"],
    "vi": ["Ngày mai thư viện sẽ đóng cửa để sửa chữa.",
           "Chúng tôi đã đi dạo quanh hồ vào buổi sáng sớm.",
           "Xin vui lòng giữ máy, nhân viên sẽ trả lời ngay."],
    "ur": ["کل صبح شہر میں بارش کا امکان ہے۔",
           "ہم نے ہفتے کے آخر میں دریا کے کنارے پکنک منائی۔"],
    "te": ["రేపు ఉదయం పాఠశాలలో వార్షికోత్సవం జరుగుతుంది.",
           "మేము వారాంతంలో సముద్ర తీరానికి వెళ్ళాము."],
    "th": ["พรุ่งนี้ห้องสมุดจะปิดเพื่อซ่อมแซม",
           "เมื่อวานเราไปเดินเล่นที่ตลาดน้ำกับครอบครัว"],
    "nl": ["De bakker op de hoek is morgen de hele dag gesloten.",
           "We hebben gisteren een lange fietstocht langs de kust gemaakt."],
    "pl": ["Jutro rano w całym mieście może padać deszcz.",
           "W weekend pojechaliśmy na wycieczkę w góry."],
    "sv": ["Tåget till Göteborg är tyvärr försenat med tjugo minuter.",
           "Vi plockade blåbär i skogen hela eftermiddagen."],
    "ko": ["내일은 비가 올 예정이니 우산을 꼭 챙기세요.",
           "주말에 친구들과 함께 바닷가에 다녀왔어요."],
    "zh": ["明天早上可能会下雨，请记得带伞。",
           "周末我们全家一起去公园野餐了。"],
    "ru": ["Завтра утром в городе ожидается сильный дождь.",
           "В выходные мы ездили к бабушке на дачу."],
    "pt": ["Amanhã de manhã vai haver obras na avenida principal.",
           "No fim de semana fomos à praia com os nossos amigos."],
}
SEALED_PLAN = {"en": 16, "es": 4, "fr": 3, "hi": 3, "de": 2, "ja": 2, "pt": 2, "ar": 2, "he": 2, "ko": 2, "zh": 2, "ru": 2,
               "ta": 3, "tl": 3}
TRAIN_PLAN = {"en": 55, "es": 28, "fr": 20, "hi": 18, "ar": 14, "he": 14, "ja": 10, "vi": 8, "ur": 6, "te": 6, "th": 6,
              "nl": 6, "pl": 5, "sv": 6, "ko": 6, "zh": 5, "ru": 5, "pt": 5}
assert not ({"ta", "tl"} & set(TRAIN_PLAN))

used = set(json.loads((ROOT / "audio/corpus3/voices.json").read_text()))
for vs in json.loads((ROOT / "audio/sealed/voices.json").read_text()).values():
    used |= set(vs)
for vs in json.loads((ROOT / "audio/corpus4/voices.json").read_text()).values():
    used |= set(vs)
used |= {r[2] for r in json.loads((ROOT / "audio/corpus5_plan.json").read_text())}
used |= {r[2] for r in json.loads((ROOT / "audio/corpus6_plan.json").read_text())}

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

rng = random.Random(2028)
sealed, train = [], []
for lang in sorted(set(SEALED_PLAN) | set(TRAIN_PLAN)):
    vs = sorted(catalogue.get(lang, [])); rng.shuffle(vs)
    ns = SEALED_PLAN.get(lang, 0); nt = TRAIN_PLAN.get(lang, 0)
    for i, v in enumerate(vs[:ns]):
        sealed.append(("sealed4", lang, v, SEALED_TEXTS[lang][i % len(SEALED_TEXTS[lang])]))
    for i, v in enumerate(vs[ns:ns + nt]):
        train.append(("corpus7", lang, v, TRAIN_TEXTS[lang][i % len(TRAIN_TEXTS[lang])]))
rng.shuffle(train)
plan = sealed + train                             # sealed set first, training voices in shuffled order
assert len({p[2] for p in plan}) == len(plan), "duplicate voice in plan"
if DRY:
    import collections
    print("sealed4", dict(collections.Counter(p[1] for p in sealed)))
    print("corpus7", dict(collections.Counter(p[1] for p in train)))
    print(f"{len(plan)} voices, ~{sum(len(t) for *_, t in plan) * len(MODELS)} characters; cap {CAP}"); raise SystemExit
for s in ("sealed4", "corpus7"):
    (ROOT / "audio" / s).mkdir(parents=True, exist_ok=True)
(ROOT / "audio/corpus7_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=0))
print(f"{len(plan)} voices planned ({len(sealed)} sealed), ~{sum(len(t) for *_, t in plan) * len(MODELS)} characters", flush=True)

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
print(f"done: ~{spent[0]} characters spent; sealed4={len(list((ROOT / 'audio/sealed4').glob('*.wav')))} "
      f"corpus7={len(list((ROOT / 'audio/corpus7').glob('*.wav')))} clips", flush=True)
