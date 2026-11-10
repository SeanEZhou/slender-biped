#!/bin/bash
# 2026-11-09 ank3: the as-designed foot (servo in the foot, straight shin) with the ankle fork angled 16 deg so the
# rest pose is the usual bent stance (hip 16 / knee 32) with the board flat. Exactly the recipe of the successful
# ankle study b150A_shift3_L (stride 20, ref_swing 6, slip -50; the x60 / x15 / -300 weights made policies yank the
# foot up and fall), per pad; then soft-landing variants (land_v) and balance.
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-50.0"
SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 ref_table="b150A_shift" settle_gate_u=[0.275,0.775]'
for pad in r80 b150 foot d70 r60; do
  EXTRA_SET="$SET" scripts/b150_run.sh a3s_${pad} shift_b150 hw_ank3_${pad}_legs_10dof 20e6 --rw $STEPRW
done
EXTRA_SET="$SET" scripts/b150_run.sh a3s_r80_soft shift_b150 hw_ank3_r80_legs_10dof 20e6 --rw $STEPRW land_v=-150.0 ang_vel_xy=-0.2
for pad in r80 d70; do
  EXTRA_SET='dr_floor_tc=[0.02,0.8] ref_table="b150A_bal1"' scripts/b150_run.sh a3b_${pad} bal_b150 hw_ank3_${pad}_legs_10dof 20e6
done
