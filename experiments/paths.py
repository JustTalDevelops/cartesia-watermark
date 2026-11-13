"""Generate the same request over SSE and WebSocket (bytes already tested) to check watermark coverage."""
import asyncio, base64, json, sys, pathlib
import numpy as np, requests, soundfile as sf, websockets
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).resolve().parent.parent / "tools")]
import cartesia as c

T = ("The orchestra tuned their instruments while the audience slowly found their seats, whispering and rustling "
     "their programmes, until the lights dimmed and the conductor finally raised her baton.")
VOICE = "47c38ca4-5f35-497b-b1a3-415245fb35e1"
FMT = {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100}
OUT = pathlib.Path(__file__).resolve().parent.parent / "audio" / "paths"


def sse():
    body = {"model_id": "sonic-3.6", "transcript": T, "voice": {"mode": "id", "id": VOICE}, "output_format": FMT, "language": "en"}
    r = requests.post(f"{c.API}/tts/sse", headers=c._headers(), json=body, stream=True, timeout=120)
    r.raise_for_status()
    chunks = []
    for line in r.iter_lines():
        if line.startswith(b"data:"):
            d = json.loads(line[5:])
            if d.get("data"):
                chunks.append(base64.b64decode(d["data"]))
    return np.frombuffer(b"".join(chunks), dtype=np.float32)


async def ws():
    url = f"wss://api.cartesia.ai/tts/websocket?api_key={c._env('CARTESIA_API_KEY')}&cartesia_version={c.VERSION}"
    chunks = []
    async with websockets.connect(url, max_size=None) as s:
        await s.send(json.dumps({"context_id": "wm1", "model_id": "sonic-3.6", "transcript": T,
                                 "voice": {"mode": "id", "id": VOICE}, "output_format": FMT, "language": "en"}))
        async for msg in s:
            d = json.loads(msg)
            if d.get("type") == "chunk":
                chunks.append(base64.b64decode(d["data"]))
            if d.get("done") or d.get("type") in ("done", "error"):
                if d.get("type") == "error":
                    print("ws error", d)
                break
    return np.frombuffer(b"".join(chunks), dtype=np.float32)


for name, fn in [("sse", sse), ("websocket", lambda: asyncio.run(ws()))]:
    x = fn(); p = OUT / f"{name}_sonic-3.6.wav"; sf.write(p, x, 44100, subtype="FLOAT")
    print(name, f"{len(x) / 44100:.1f}s", c.detect(p), flush=True)
