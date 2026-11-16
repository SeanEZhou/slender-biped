#!/bin/bash
# round 17 final ranking (after robot17_queue6.sh): gait_eval with 96 runs, a fresh seed, robot and loose sims, plus the
# faster command (0.03 m/s) on the robot sim. One process at a time.
cd "$(dirname "$0")/.."
until grep -q "queue6 done" runs/robot17_queue6.out; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
out=results/robot17/final_rank.txt; : > $out
for r in k5_high2:params_0016M.pkl m2_attland l4_mid l7_s15 l9_s18 l13_lv2; do
  for s in robot loose; do $G $r --n 96 --seed 7 --sim $s 2>&1 | grep -A1 "^gait" >> $out; done
  echo "$(date) ranked $r"
done
for r in m2_attland l4_mid l7_s15 l9_s18 l13_lv2; do
  $G $r --n 96 --seed 7 --vx 0.03 2>&1 | grep -A1 "^gait" >> $out; echo "$(date) fast $r"
done
echo "$(date) final done"
