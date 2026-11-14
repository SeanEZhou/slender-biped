#!/bin/bash
# round 9 follow-up, after robot9_queue.sh: d6_glcp cut the gyro gain 80% (0.011 / 0.016) but walks 21 cm and leans
# forward: try a fifth of the penalty. d9_blind was still learning (episodes 385 steps): 28M more from its end.
cd "$(dirname "$0")/.."; . scripts/robot9_common.sh
while ps -eo args | grep -q "^/bin/bash scripts/robot9_queue.sh"; do sleep 30; done
br d6b_glcp01 "$R05" --lcp 0.01 --lcp-idx 3:6
scripts/robot9_analysis/screen9.sh d6b_glcp01 > /dev/null 2>&1 &
[ $(date +%H%M) -lt 0915 ] && br d9c_blind "$R05 gyro_obs_scale=0.0" --init runs/d9_blind/params.pkl
scripts/robot9_analysis/screen9.sh d9c_blind > /dev/null 2>&1
