#!/bin/bash
# Round 16: once m1 is done, the walk tests one at a time (no lane training meanwhile), then m2 may start.
cd "$(dirname "$0")/.."
while [ ! -f runs/m1_attgate/done.json ]; do sleep 20; done
o=results/robot16/eval_walk.txt; : > $o
for sim in loose robot; do
  for spec in "k5_high2:params_0016M.pkl|" "k5_high2:params_0016M.pkl|settle_gate_att=[0.052,0.105]" "m1_attgate|"; do
    run=${spec%%|*}; set_=${spec#*|}
    echo "== $sim $run ${set_:-(as trained)}" >> $o
    timeout 900 scripts/gpu_python.sh scripts/push_eval.py walk $run --no-sup --sim $sim --n 16 --dv 0.0 0.1 ${set_:+--set $set_} 2>&1 \
      | grep -E "^ *0\.[0-9]+ \||overall|walk test" >> $o
  done
done
touch runs/robot16_eval1.done
