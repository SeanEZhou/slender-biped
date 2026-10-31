#!/bin/bash
# Round-5 screening: when a run finishes, sideways drift (20 seeds, heading hold, 0.15 m/s) of the rule checkpoint and
# the final one, plus gait quality (3 seeds) of the final one. Usage: scripts/screen_r5.sh run1 run2 ...
# Appends to results/drift/screen.txt.
cd "$(dirname "$0")/.."
mkdir -p results/drift
for r in "$@"; do
  until [ -f runs/$r/done.json ] || ! { pgrep -f "name $r " >/dev/null || grep -q -- "--name $r " runs/queue.txt; }; do sleep 30; done
  [ -f runs/$r/done.json ] || { echo "$r did not finish" | tee -a results/drift/screen.txt; continue; }
  P=$(.venv/bin/python scripts/best_checkpoint.py $r)
  last=$(ls runs/$r/params_*M.pkl | tail -1 | xargs basename)
  for c in $(echo "$P $last" | tr ' ' '\n' | sort -u); do
    .venv/bin/python scripts/drift_eval.py $r --params $c 2>&1 | grep -v -i warp | tee -a results/drift/screen.txt
  done
  .venv/bin/python scripts/gait_quality.py $r --params $last --seeds 3 2>&1 | grep -v -i warp | tee -a results/drift/screen.txt
done
