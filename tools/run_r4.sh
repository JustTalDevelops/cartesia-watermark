#!/usr/bin/env bash
# Round 4 pipeline (unattended): XLS-R layer-2 features -> contrastive variants -> 3 plain + 3 contrastive heads
# -> calibrate on validation (plain vs contrastive) -> score sealed4 exactly once when Cartesia's labels are complete.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export FEAT=xlsr2
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for XLS-R feature extraction"
until [ "$(grep -l 'done' results/framesx_*.log 2>/dev/null | wc -l)" -eq 6 ]; do sleep 30; done
log "contrastive variants"
for i in 0 1 2 3 4 5; do $PY -u tools/frames_contrast.py $i 6 4 > results/framesxc_$i.log 2>&1 & done
wait

log "training heads (one at a time)"
for s in 0 1 2; do
  for kind in ls c; do
    tag=x4$kind$s; [ -f results/head_conv_heldout_$tag.pt ] && continue
    if [ $kind = c ]; then export CONTRAST=1; else export CONTRAST=0; fi
    SEED=$s SMOOTH=0.1 TAG=_$tag THREADS=22 $PY -u tools/train_head.py conv heldout 6 > results/head_conv_$tag.log 2>&1
    echo "$tag: $(grep -h -E 'AUC|FPR' results/head_conv_$tag.log | tr '\n' ' ')"
  done
done

log "calibrating on validation"
$PY -u tools/calibrate_final.py results/final_detector_r4.json x4ls0,x4ls1,x4ls2 x4c0,x4c1,x4c2 2>&1 | grep -v -E "Warning|Loading"

log "waiting for Cartesia labels on sealed4"
until [ "$($PY -c 'import json;print(len(json.load(open("results/sealed4_cartesia.json"))))' 2>/dev/null || echo 0)" -ge 192 ]; do sleep 60; done

log "scoring sealed4 (once)"
$PY -u tools/sealed_final.py sealed4 results/final_detector_r4.json results/sealed4_cartesia.json \
  en,es,fr,de,ja,hi,pt,it,ko,zh,nl,ru,pl,sv,tr,he,ar 2>&1 | grep -v -E "Warning|Loading"
log "round 4 done"
