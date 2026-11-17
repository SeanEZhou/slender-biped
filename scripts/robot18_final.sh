#!/bin/bash
# round 18 final ranking (after robot18_queue9.sh): 96 walks per row, a fresh seed (7), robot + loose sims, the same
# for every policy; l9 (the start) and n7 / n11 (the bases) against the late lanes and their 10M checkpoints.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until grep -q "queue10 done" runs/robot18_queue10.out; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
for r in l9_s18 n7_s12 n11_s16r n14_s19 n15_s20 n15_s20:params_0010M.pkl n16_s21 n16_s21:params_0010M.pkl n18_s23 n18_s23:params_0010M.pkl n17_s24 n19_s25 n19_s25:params_0010M.pkl; do
  d=${r%%:*}; [ -f runs/$d/done.json ] || continue
  for sim in robot loose; do $G $r --n 96 --seed 7 --sim $sim --profile; done
done 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot18/final_rank.txt
echo "$(date) final done"
