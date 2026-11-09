#!/bin/bash
# 2026-11-09: after shift_queue2.sh: the as-built walker from its best stepper (b150_shift2_L 10 M) with falling made
# expensive (termination -100) and the stepping rewards between the slide-prone and the fall-prone settings
cd "$(dirname "$0")/.."
while ps -eo args | grep -v grep | grep -q "scripts/shift_queue2.sh"; do sleep 30; done
EXTRA_SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005' scripts/b150_run.sh b150_shift4_L shift_b150 hw_pad_board150w_legs_8dof 20e6 \
  --init runs/b150_shift2_L/params_0010M.pkl --rw progress=0.0 stride=40.0 ref_swing=10.0 slip=-300.0 termination=-100.0
