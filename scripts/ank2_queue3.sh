#!/bin/bash
# 2026-11-09 round 3: a2t_* (ref_swing x15, termination -20) learned to yank the swing foot up and fall (episodes got
# shorter with training). Balance first, then step with survival weighted over lifting.
cd "$(dirname "$0")/.."
EXTRA_SET='dr_floor_tc=[0.02,0.8]' scripts/b150_run.sh a2b_r80 ank2_bal hw_ank2_r80_legs_10dof 20e6 --rw slip=-100.0 termination=-50.0 alive=2.0
STEPRW="progress=0.0 stride=30.0 ref_swing=4.0 slip=-100.0 termination=-50.0 alive=2.0"
for pad in r80 b150; do
  EXTRA_SET='dr_floor_tc=[0.02,0.8]' scripts/b150_run.sh a2u_${pad} ank2_step hw_ank2_${pad}_legs_10dof 20e6 --rw $STEPRW
done
