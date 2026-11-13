#!/usr/bin/env bash
# Round 3 pipeline (unattended), replacing run_r3.sh's fine-tune step with mark-targeted contrastive variants:
# contrast features -> 3 contrastive heads -> calibrate (r3 frozen vs r3 contrastive, validation only)
# -> score sealed3 exactly once when Cartesia's labels are complete.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for contrastive feature extraction"
until [ "$(grep -l 'done' results/frames_contrast_*.log 2>/dev/null | wc -l)" -eq 6 ]; do sleep 30; done

log "training 3 contrastive conv heads (one at a time: three in parallel thrash 60 GB RAM)"
for s in 0 1 2; do
  [ -f results/head_conv_heldout_r3c$s.pt ] && continue
  CONTRAST=1 SEED=$s SMOOTH=0.1 TAG=_r3c$s THREADS=22 $PY -u tools/train_head.py conv heldout 6 > results/head_conv_r3c$s.log 2>&1
  grep -h -E "AUC|FPR|probe" results/head_conv_r3c$s.log || true
done

log "calibrating on validation"
$PY -u tools/calibrate_final.py results/final_detector_r3.json r3ls0,r3ls1,r3ls2 r3c0,r3c1,r3c2 2>&1 | grep -v -E "Warning|Loading"

log "waiting for Cartesia labels on sealed3"
until [ "$($PY -c 'import json;print(len(json.load(open("results/sealed3_cartesia.json"))))' 2>/dev/null || echo 0)" -ge 160 ]; do sleep 60; done

log "scoring sealed3 (once)"
$PY -u tools/sealed_final.py sealed3 results/final_detector_r3.json results/sealed3_cartesia.json \
  en,es,fr,de,ja,hi,pt,it,ko,zh,nl,ru,pl,sv,tr 2>&1 | grep -v -E "Warning|Loading"
log "round 3 done"
