#!/bin/bash
# Round 16, 23:16: m2_attland was best in both walk tests at 14M (6M less than m1): 10M more, alone, then its tests.
cd "$(dirname "$0")/.."
source scripts/robot16_common.sh
INIT=runs/m2_attland/params_0014M.pkl
b16 m2b_attland 10e6 "$ATT" foot_tilt=-30.0
o=results/robot16/eval_walk.txt
for sim in loose robot; do
  echo "== $sim m2b_attland (m2 + 10M)" >> $o
  timeout 900 scripts/gpu_python.sh scripts/push_eval.py walk m2b_attland --no-sup --sim $sim --n 16 --dv 0.0 0.1 2>&1 \
    | grep -E "^ *0\.[0-9]+ \||overall|walk test" >> $o
done
echo "$(date) lane3 done"
