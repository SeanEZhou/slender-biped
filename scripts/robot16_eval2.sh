#!/bin/bash
# Round 16: m2's walk tests once it is done (alone).
cd "$(dirname "$0")/.."
while [ ! -f runs/m2_attland/done.json ]; do sleep 20; done
o=results/robot16/eval_walk.txt
for sim in loose robot; do
  echo "== $sim m2_attland (as trained)" >> $o
  timeout 900 scripts/gpu_python.sh scripts/push_eval.py walk m2_attland --no-sup --sim $sim --n 16 --dv 0.0 0.1 2>&1 \
    | grep -E "^ *0\.[0-9]+ \||overall|walk test" >> $o
done
touch runs/robot16_eval2.done
