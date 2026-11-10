#!/bin/bash
# 2026-11-09 after ank3_queue2: longer training for the short pads, soft-landing fine-tune of the best (b150), and a
# fair retry of the straight-leg design (ank2) with the recipe that works (stride 20, ref_swing 6, slip -50).
cd "$(dirname "$0")/.."
while pgrep -f "ank3_queue2.sh" > /dev/null || pgrep -f "ank3_queue.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-50.0"
SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 ref_table="b150A_shift" settle_gate_u=[0.275,0.775]'
EXTRA_SET="$SET" scripts/b150_run.sh a3s_b150_soft shift_b150 hw_ank3_b150_legs_10dof 15e6 --init runs/a3s_b150/params_0020M.pkl --rw $STEPRW land_v=-150.0 ang_vel_xy=-0.2
EXTRA_SET="$SET" scripts/b150_run.sh a3s_foot2 shift_b150 hw_ank3_foot_legs_10dof 20e6 --init runs/a3s_foot/params_0020M.pkl --rw $STEPRW
EXTRA_SET="$SET" scripts/b150_run.sh a3s_r80_2 shift_b150 hw_ank3_r80_legs_10dof 20e6 --init runs/a3s_r80/params_0020M.pkl --rw $STEPRW
EXTRA_SET='dr_floor_tc=[0.02,0.8]' scripts/b150_run.sh a2f_b150 ank2_step hw_ank2_b150_legs_10dof 20e6 --rw $STEPRW
