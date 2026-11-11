#!/bin/bash
# 2026-11-10 round 3: c2_slow (foot_tilt -30) collapsed to falling every episode by 10 M; the spectrum again with
# foot_tilt -10 (touchdown tilt is still penalized, toe-down x2, but no longer outweighs staying up).
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0"
CLEAN="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-0.5 ang_acc=-0.003 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
B='dr_floor_tc=[0.02,0.8]'
EXTRA_SET="$B ref_swing_h=0.010 ref_table=\"c_T12\" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04" \
  scripts/b150_run.sh c3_T12 shift_b150 hw_ank3_b150_legs_10dof 25e6 --init runs/a3s_b150/params_0020M.pkl --rw $STEPRW $CLEAN
EXTRA_SET="$B ref_swing_h=0.010 ref_table=\"c_T08\" gait_hz=0.625 settle_gate_u=[0.1875,0.6875] cmd_vx=[0.015,0.04] stride_target=0.04" \
  scripts/b150_run.sh c4_T08 shift_b150 hw_ank3_b150_legs_10dof 25e6 --init runs/a3s_b150/params_0020M.pkl --rw $STEPRW $CLEAN
EXTRA_SET="$B ref_swing_h=0.015 ref_table=\"c_big\" gait_hz=0.25 settle_gate_u=[0.225,0.725] cmd_vx=[0.01,0.03] stride_target=0.06" \
  scripts/b150_run.sh c5_big shift_b150 hw_ank3_b150_legs_10dof 25e6 --init runs/a3s_b150/params_0020M.pkl --rw $STEPRW $CLEAN
EXTRA_SET="$B ref_swing_h=0.005 ref_table=\"b150A_slow1\" gait_hz=0.125 settle_gate_u=[0.1875,0.6875]" \
  scripts/b150_run.sh c2_slow2 shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/a3w_b150/params_0020M.pkl --rw $STEPRW $CLEAN
