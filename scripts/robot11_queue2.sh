#!/bin/bash
# Round 11, second part (19:40): p2 (sideways open loop + roll gate + loose hips, from k5) never learned to stay up
# (episodes 128 -> 105 steps in 24M). Same with round 10's fitted hip roll, to see whether the loose hips block it.
cd "$(dirname "$0")/.."
source scripts/robot11_common.sh
SER_FIT='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
while ps -eo args | grep -qE "^/bin/bash scripts/robot11_queue.sh|^[^ ]*python[^ ]* scripts/train.py"; do sleep 20; done
SER=$SER_FIT br p3_side_fit 20e6 "$SIDE $RGATE"
echo "$(date) queue2 done"
