#!/bin/bash
# Round 13 queue 2: a recovery specialist trained on k5's own tipping states, and an earlier-firing supervisor.
cd "$(dirname "$0")/.."
source scripts/robot13_common.sh
while ps -eo args | grep -qE "^[^ ]*bash scripts/robot13_queue.sh" || ps -eo args | grep -qE "^[^ ]*python[^ ]* scripts/train.py"; do sleep 30; done
WALKER='sup_walker="runs/k5_high2/params_0016M.pkl" sup_walker_lpf=0.7'
br r13e_spec      30e6 "cmd_zero_prob=0.0 push_dv=[0.05,0.3] $SUP $WALKER"   # frozen k5 walks (filtered as on the robot); this policy only recovers
br r13c2_sup_early 24e6 "push_dv=[0.03,0.25] sup=true sup_roll=0.175 sup_pitch=0.175 sup_tau=0.15"   # supervisor at 10 / 10 deg, tau 0.15: 0.4 s more warning, more false stops
echo "$(date) queue done"
