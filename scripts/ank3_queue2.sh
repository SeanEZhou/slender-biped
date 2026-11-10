#!/bin/bash
# 2026-11-09 ank3 slow, soft steps (the user: lift, swing while balancing on one foot, lower slowly): b150A_slow1
# (4 s per step: shift 1.2 s, held through a 1.2 s swing that lifts over 35 % and lowers over 65 %, released 0.8 s
# after touchdown), with the soft-landing penalty.
cd "$(dirname "$0")/.."
while pgrep -f "ank3_queue.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-50.0 land_v=-150.0 ang_vel_xy=-0.2"
SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 ref_table="b150A_slow1" gait_hz=0.125 settle_gate_u=[0.1875,0.6875]'
EXTRA_SET="$SET" scripts/b150_run.sh a3w_r80 shift_b150 hw_ank3_r80_legs_10dof 30e6 --init runs/a3s_r80/params_0020M.pkl --rw $STEPRW
EXTRA_SET="$SET" scripts/b150_run.sh a3w_b150 shift_b150 hw_ank3_b150_legs_10dof 30e6 --rw $STEPRW
