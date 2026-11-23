#!/bin/bash
# round 29 queue 4 (03:55). The walking evals sampled the reset command with the run's turn_prob, so 30 % of z9_t1's
# walkers turned in place and stood (fixed: turn_prob=0 in gait_eval / startstop / limp_sim, 1e-6 in turn_eval):
# z9_t1 again. z9_t1 turns (45 deg: 41-44, 90: 72-92) and stays up through 75-94 % of the turns.
cd "$(dirname "$0")/.."; source scripts/robot29_common.sh
until grep -q "queue3 done" runs/robot29_queue.log; do sleep 20; done
late() { [ $(date +%H%M) -gt 0845 ] && [ $(date +%H%M) -lt 1200 ] && { echo "$(date) past 08:45, not starting $1"; return 0; }; return 1; }
G="scripts/gpu_python.sh scripts/gait_eval.py"
{ for x in 0.015 0.035; do $G z0_y2c --n 48 --vx $x --set "$NG"; done; $G z0_y2c --n 48 --vx 0.025 --sim loose --set "$NG"
  scripts/gpu_python.sh scripts/robot27_analysis/limp_sim.py z0_y2c; scripts/gpu_python.sh scripts/robot27_analysis/limp_sim.py z0_y2c --nogate; } 2>&1 | grep -vi "$FLT" > results/robot29/base_z0_y2c.txt
mv results/robot29/eval_z9_t1.txt results/robot29/turn_z9_t1.txt results/robot29/old/ 2>/dev/null
ev29 z9_t1; evturn z9_t1
INIT=runs/z9_a5m/params_0020M.pkl
late z9_t2 || L29t z9_t2 30e6 "$START $TURNY" land_v=-100.0 $LATE --anchor 5.0 --mirror 1.0     # turning from z9_a5m
INIT=runs/z0_y2c/params_0020M.pkl
late z9_as5 || L29 z9_as5 20e6 "$START $ANCW $STAG" land_v=-100.0 $LATE --anchor 5.0 --mirror 1.0   # + z7_anc2's steadiness
echo "$(date) queue4 done"
