from pathlib import Path
from functools import wraps
import hashlib,json,sys,time
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));from experiments.score_codec_study import TextScorer,digest
from speech_repair import units,disagreement
import torch
D=ROOT/".cache/removal-reliability/codec_fresh";OUT=ROOT/".cache/removal-reliability/mms_bf16_originals";OUT.mkdir(exist_ok=False)
plan=json.loads((D/"evaluation.plan.json").read_text());baseline=json.loads((D/"scores/originals_mms.json").read_text());base_config=hashlib.sha256(json.dumps(baseline["metric"],sort_keys=True).encode()).hexdigest()
cache=[json.loads(p.read_text()) for p in (D/"scores/cache").glob("*.json")]
scorer=TextScorer("mms",OUT/"cache");scorer.cache.mkdir();scorer.metadata["precision"]="bf16-autocast-forward";scorer.configuration=hashlib.sha256(json.dumps(scorer.metadata,sort_keys=True).encode()).hexdigest()
original_forward=scorer.model.forward
@wraps(original_forward)
def mixed(*args,**kwargs):
    with torch.autocast("cpu",dtype=torch.bfloat16):return original_forward(*args,**kwargs)
scorer.model.forward=mixed
report={"source_sha256":digest(Path(__file__)),"role":"Original-only precision diagnostic; no candidate outputs or frozen quality scores changed","fresh_originals_read":16,"candidate_outputs_read":False,"model":scorer.metadata,"clips":[],"complete":False}
for item in plan["items"]:
    source=ROOT/item["source"];assert digest(source)==item["sha256"]
    matches=[c for c in cache if c.get("sha256")==item["sha256"] and c.get("configuration_sha256")==base_config];assert len(matches)==1
    started=time.perf_counter();text=scorer.transcribe(source,item["language"]);elapsed=time.perf_counter()-started
    before=units(matches[0]["transcript"],scorer.normalize,item["language"]);after=units(text,scorer.normalize,item["language"])
    row={"id":item["id"],"source_sha256":item["sha256"],"exact_normalized_unit_parity":before==after,"disagreement_from_fp32":disagreement(before,after),"inference_seconds_including_preprocessing":elapsed}
    report["clips"].append(row);(OUT/"result.json").write_text(json.dumps(report,indent=2));print(row,flush=True)
report["complete"]=True;(OUT/"result.json").write_text(json.dumps(report,indent=2));print("MMS_BF16_ORIGINAL_DIAGNOSTIC_COMPLETE",flush=True)
