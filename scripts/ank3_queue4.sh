#!/bin/bash
# 2026-11-09 slow soft step, fine-tune: a3w_b150 at 10 M landed softly (30 mm/s) but fell now and then; at 20 M it
# never fell but landed at ~90 mm/s. Fine-tune the 20 M policy with a much stronger touchdown-speed penalty.
cd "$(dirname "$0")/.."
while pgrep -f "ank3_queue3.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-50.0 land_v=-600.0 ang_vel_xy=-0.3"
SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 ref_table="b150A_slow1" gait_hz=0.125 settle_gate_u=[0.1875,0.6875]'
EXTRA_SET="$SET" scripts/b150_run.sh a3x_b150 shift_b150 hw_ank3_b150_legs_10dof 15e6 --init runs/a3w_b150/params_0020M.pkl --rw $STEPRW
