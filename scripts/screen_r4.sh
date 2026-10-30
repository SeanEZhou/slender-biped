#!/bin/bash
# Round-4 screening: when a run finishes, gait quality (3 seeds, 0.15 m/s) of the rule checkpoint and the final one.
# Usage: scripts/screen_r4.sh run1 run2 ...   (appends to results/gait/screen.txt)
cd "$(dirname "$0")/.."
mkdir -p results/gait
for r in "$@"; do
  until [ -f runs/$r/done.json ] || ! { pgrep -f "name $r " >/dev/null || grep -q -- "--name $r " runs/queue.txt; }; do sleep 30; done
  [ -f runs/$r/done.json ] || { echo "$r did not finish" | tee -a results/gait/screen.txt; continue; }
  P=$(.venv/bin/python scripts/best_checkpoint.py $r)
  last=$(ls runs/$r/params_*M.pkl | tail -1 | xargs basename)
  for c in $(echo "$P $last" | tr ' ' '\n' | sort -u); do
    .venv/bin/python scripts/gait_quality.py $r --params $c --seeds 3 2>&1 | grep -v -i warp | tee -a results/gait/screen.txt
  done
done
