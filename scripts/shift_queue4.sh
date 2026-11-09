#!/bin/bash
# 2026-11-09: after shift_queue3.sh: the ankle-roll walker (b150A_shift3_L 10 M: 6+ steps in 16/16 hard-floor walks)
# on softer rugs, and with torso and head on
cd "$(dirname "$0")/.."
while ps -eo args | grep -v grep | grep -q "scripts/shift_queue3.sh"; do sleep 30; done
SET='ref_swing_h=0.005 ref_table="b150A_shift" settle_gate_u=[0.275,0.775]'
RW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-50.0"
EXTRA_SET="dr_floor_tc=[0.1,1.2] $SET" scripts/b150_run.sh b150A_shift3s_L shift_b150 hw_pad_board150w_legs_ankle_10dof 20e6 \
  --init runs/b150A_shift3_L/params_0010M.pkl --rw $RW
EXTRA_SET="dr_floor_tc=[0.02,0.8] $SET" scripts/b150_run.sh b150A_shift_F shift_b150 hw_pad_board150w_ankle_10dof 30e6 \
  --init runs/b150A_shift3_L/params_0010M.pkl --rw $RW
