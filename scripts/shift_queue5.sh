#!/bin/bash
# 2026-11-09: after shift_queue4.sh: the ankle-roll walker with robot-realistic sensing (IMU filter, +-3 deg tilt
# reading offset), from b150A_shift3_L 10 M
cd "$(dirname "$0")/.."
while ps -eo args | grep -v grep | grep -q "scripts/shift_queue4.sh"; do sleep 30; done
EXTRA_SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 ref_table="b150A_shift" settle_gate_u=[0.275,0.775] imu_filter=true imu_gate=0.15 gyro_range=17.45 dr_imu_tilt=0.05' \
  scripts/b150_run.sh b150A_shift3i_L shift_b150 hw_pad_board150w_legs_ankle_10dof 20e6 \
  --init runs/b150A_shift3_L/params_0010M.pkl --rw progress=0.0 stride=20.0 ref_swing=6.0 slip=-50.0
