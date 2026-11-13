"""Generate paired sonic-3.6 / sonic-3.5 corpus with identical requests."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cartesia as c

TEXTS = [
    "In the early morning, the harbour was quiet except for the gulls circling over the fishing boats. "
    "Old Thomas checked his nets one last time, then pushed off into the grey water, humming a song his father taught him.",
    "Quarterly revenue grew eleven percent year over year, driven primarily by strong demand in the enterprise segment. "
    "Operating margins expanded as we continued to invest in automation and reduce our infrastructure costs.",
    "To reset your password, open the settings menu, select account, and tap security. "
    "You will receive a six digit verification code by text message. Enter the code within five minutes to continue.",
    "Photosynthesis converts light energy into chemical energy stored in glucose. "
    "Chlorophyll absorbs mostly blue and red wavelengths, which is why most leaves appear green to the human eye.",
]
VOICES = {"daniel": "47c38ca4-5f35-497b-b1a3-415245fb35e1", "jacqueline": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"}
OUT = pathlib.Path(__file__).resolve().parent.parent / "audio" / "corpus"
OUT.mkdir(parents=True, exist_ok=True)
for ti, t in enumerate(TEXTS):
    for vn, vid in VOICES.items():
        for m in ["sonic-3.6", "sonic-3.5"]:
            p = OUT / f"t{ti}_{vn}_{m}.wav"
            if not p.exists():
                c.tts(t, p, model=m, voice=vid)
                print("wrote", p.name)
