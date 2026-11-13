"""Large training corpus: ~100 new voices x 8 texts x 4 models (voices disjoint from corpus3 and sealed).
(The account ran out of credits partway: 782 clips, 26 complete English voices.)"""
import json, sys, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c, requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "audio" / "corpus4"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["sonic-3.6", "sonic-preview", "sonic-3.5", "sonic-3"]
TEXTS = {
    "en": ["The bakery on the corner opens before sunrise, and by seven there is always a queue of sleepy people outside.",
           "Please hold while I transfer your call to the billing department. A representative will be with you shortly.",
           "Most of the volcano's eruptions happened thousands of years ago, but scientists still monitor it every single day.",
           "I left my umbrella at the restaurant again, so of course it started raining the moment I stepped outside.",
           "The final exam will cover chapters four through nine, with particular emphasis on the second half of the course.",
           "After three days of hiking through the forest, they finally reached the ridge and saw the lake far below them.",
           "Thank you for your patience. Your refund has been processed and should appear in your account within a week.",
           "He tuned the old radio slowly until, through the static, a faint voice began reading the evening news."],
    "de": ["Der Zug nach Hamburg hat heute leider etwa fünfzehn Minuten Verspätung wegen einer Signalstörung.",
           "Am Wochenende wollen wir mit den Kindern in den Bergen wandern, wenn das Wetter einigermaßen mitspielt.",
           "Bitte vergessen Sie nicht, Ihre Unterlagen bis spätestens Freitag beim Sekretariat abzugeben.",
           "Die Ausstellung im Stadtmuseum zeigt Fotografien aus den zwanziger Jahren, die bisher nie veröffentlicht wurden.",
           "Ich habe gestern Abend ein neues Rezept ausprobiert, und ehrlich gesagt war es überraschend lecker.",
           "Unser Kundenservice ist von Montag bis Freitag zwischen acht und achtzehn Uhr telefonisch erreichbar.",
           "Im Herbst färben sich die Blätter im Park rot und gelb, und die Luft riecht nach Regen und Erde.",
           "Wenn Sie Fragen zu Ihrer Rechnung haben, wenden Sie sich bitte direkt an unsere Buchhaltung."],
    "pt": ["O mercado da cidade abre cedo, e às sete da manhã já está cheio de gente comprando frutas e pão fresco.",
           "Por favor, confirme o seu endereço de e-mail antes de continuar com o cadastro no nosso sistema.",
           "Durante as férias, visitamos uma pequena vila no litoral onde o tempo parecia passar mais devagar.",
           "A reunião de amanhã foi adiada para a próxima semana por causa da viagem do diretor.",
           "Minha avó sempre dizia que um bom café precisa de paciência e de água na temperatura certa.",
           "O museu vai receber uma nova exposição sobre a história da navegação no século dezesseis.",
           "Se tiver alguma dúvida sobre o seu pedido, entre em contato com o nosso atendimento ao cliente.",
           "Depois da chuva, as crianças correram para a rua para brincar nas poças e ver o arco-íris."],
    "it": ["Il treno per Milano partirà con venti minuti di ritardo a causa di un guasto tecnico sulla linea.",
           "Domenica andremo al mare con i nonni, se il tempo sarà bello come dicono le previsioni.",
           "Si prega di presentare tutti i documenti richiesti entro la fine del mese presso l'ufficio anagrafe.",
           "La nuova mostra al museo racconta la storia della città attraverso fotografie e lettere private.",
           "Ieri sera ho provato a fare la pasta in casa, ed è venuta molto meglio di quanto pensassi.",
           "Il nostro servizio clienti è disponibile dal lunedì al venerdì, dalle nove del mattino alle sei di sera.",
           "In autunno le colline si colorano di rosso e di oro, e l'aria profuma di vino e di legna.",
           "Per qualsiasi domanda sulla fattura, la invitiamo a contattare direttamente il nostro ufficio contabilità."],
}
exclude = set(json.loads((ROOT / "audio/corpus3/voices.json").read_text()))
for vs in json.loads((ROOT / "audio/sealed/voices.json").read_text()).values():
    exclude |= set(vs)


def voices(lang, n):
    r = requests.get(f"{c.API}/voices", headers=c._headers(), params={"limit": 100, "language": lang}, timeout=30).json()
    return [v["id"] for v in r["data"] if v["id"] not in exclude][:n]


V = {"en": voices("en", 58)}
for lang in ("de", "pt", "it"):
    V[lang] = voices(lang, 14)
(OUT / "voices.json").write_text(json.dumps(V))
print({k: len(v) for k, v in V.items()}, flush=True)
jobs = [(lang, vi, ti, m) for lang in V for vi in range(len(V[lang])) for ti in range(8) for m in MODELS]


def gen(j):
    lang, vi, ti, m = j
    p = OUT / f"{lang}{ti}_{lang}{vi:02d}_{m}.wav"
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
