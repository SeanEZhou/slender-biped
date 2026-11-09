#!/bin/bash
# 2026-11-09 queue, one run at a time (WSL has 7.9 GB).
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=60.0 ref_swing=15.0 slip=-300.0"   # slip -2000: slip dominated, falling early paid
# ankle roll, gentler shift (b150A_shift: 0.16 rad over 1.2 s; lift-offs at 0.281 / 0.781 of the cycle)
EXTRA_SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 ref_table="b150A_shift" settle_gate_u=[0.275,0.775]' \
  scripts/b150_run.sh b150A_shift2_L shift_b150 hw_pad_board150w_legs_ankle_10dof 30e6 --rw $STEPRW
# one-foot balance, as built
EXTRA_SET='dr_floor_tc=[0.02,0.8]' scripts/b150_run.sh b150_bal_L bal_b150 hw_pad_board150w_legs_8dof 30e6
# as built, robot-realistic sensing: the IMU filter and a +-3 deg tilt-reading offset, from the best stepper so far
EXTRA_SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 imu_filter=true imu_gate=0.15 gyro_range=17.45 dr_imu_tilt=0.05' \
  scripts/b150_run.sh b150_shift3_L shift_b150 hw_pad_board150w_legs_8dof 30e6 --init runs/b150_shift2_L/params_0010M.pkl --rw $STEPRW
# one-foot balance with ankle roll
EXTRA_SET='dr_floor_tc=[0.02,0.8] ref_table="b150A_bal1"' scripts/b150_run.sh b150A_bal_L bal_b150 hw_pad_board150w_legs_ankle_10dof 30e6
