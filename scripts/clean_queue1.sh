#!/bin/bash
# 2026-11-10 clean steps, round 1: the best quick-step policy (a3s_b150) fine-tuned with penalties on body wobble
# (ang_vel_xy, ang_acc), jerky actions (action_rate, action_acc), scuffing near the floor, tilted feet near the floor,
# planted-foot slide and too little swing clearance.
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0"
CLEAN="scuff=-100.0 foot_tilt=-30.0 ang_vel_xy=-0.5 ang_acc=-0.003 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 ref_table="b150A_shift" settle_gate_u=[0.275,0.775]'
EXTRA_SET="$SET" scripts/b150_run.sh c1_b150 shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/a3s_b150/params_0020M.pkl --rw $STEPRW $CLEAN
