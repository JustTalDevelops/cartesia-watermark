"""Check repeated removal passes against the original audio's quality budget."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import soundfile as sf
import torch
from remover import ROOT, load_model, process_audio, quality_metrics
from eval_remover import speech_metrics


def main():
    torch.set_num_threads(2)
    model, checkpoint = load_model(ROOT / "results/remover_oracle_wave5.pt")
    out = ROOT / "audio/multipass_probe"
    out.mkdir(exist_ok=True)
    plan = json.loads((ROOT / "audio/oracle_round1/plan.json").read_text())
    rows = []
    for row in plan:
        x, sr = sf.read(ROOT / row["source"])
        y = x.copy()
        for count in range(1, 5):
            y = process_audio(model, y, sr)
            metrics = {**quality_metrics(x, y), **speech_metrics(x, y, sr)}
            output = out / f"{row['language']}_pass{count}.wav"
            sf.write(output, y, sr, subtype="FLOAT")
            item = {"source": row["source"], "output": str(output.relative_to(ROOT)),
                    "language": row["language"], "voice": row["voice"], "passes": count, **metrics}
            rows.append(item)
            print(row['language'],count,metrics,flush=True)
        (ROOT / "results/multipass_probe.json").write_text(json.dumps({"checkpoint_step":checkpoint["step"],"clips":rows},indent=2))


if __name__ == "__main__":
    main()
