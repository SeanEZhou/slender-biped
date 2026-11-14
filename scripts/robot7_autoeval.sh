#!/bin/bash
# Round 7: evaluate each run on the ringing simulators when it finishes (9 x 30 s each, 3 processes while training).
cd "$(dirname "$0")/.."
for r in ${RUNS:-r1_raw r3_hist r1b_raw r7_h4 r5_histlpf}; do
  until [ -f runs/$r/done.json ]; do sleep 60; done
  for s in r7 r7hard; do
    EVAL_PROCS=3 .venv/bin/python scripts/eval6.py $r --n 9 --sim $s 2>&1 | grep -v -i warp >> results/robot7/eval_$r.txt
  done
  echo "$(date +%H:%M) evaluated $r" >> results/robot7/autoeval.out
done
