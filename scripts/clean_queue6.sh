#!/bin/bash
# 2026-11-10 round 6: variants of the continuous walker (c3i_T12, the best so far): a calmer 1.5 s step (c_T15), more
# swing clearance (c_T12h: 0.6 rad knee + hip lift, against toe catching), and random pushes in training (c3p).
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/clean_queue5.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
CLEAN="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-0.5 ang_acc=-0.003 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
B='dr_floor_tc=[0.02,0.8]'
EXTRA_SET="$B ref_swing_h=0.010 ref_table=\"c_T15\" gait_hz=0.3333333 settle_gate_u=[0.2167,0.7167] cmd_vx=[0.01,0.03] stride_target=0.04" \
  scripts/b150_run.sh c6_T15 shift_b150 hw_ank3_b150_legs_10dof 25e6 --init runs/c3i_T12/params_0025M.pkl --rw $STEPRW $CLEAN
EXTRA_SET="$B ref_swing_h=0.015 ref_table=\"c_T12h\" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04" \
  scripts/b150_run.sh c7_T12h shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/c3i_T12/params_0025M.pkl --rw $STEPRW $CLEAN
EXTRA_SET="$B ref_swing_h=0.010 ref_table=\"c_T12\" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04" \
  scripts/b150_run.sh c3p_T12 shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/c3i_T12/params_0025M.pkl --push 0.05 --rw $STEPRW $CLEAN
