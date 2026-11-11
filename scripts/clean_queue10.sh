#!/bin/bash
# 2026-11-10 round 10: (1) c_big2, between the big step and the 1.5 s step: 1.6 s per step, 0.6 rad lift, shift held
# through the swing, up to 6 cm, from the big-step policy (c5i_big); (2) the 1.5 s walker (c6_T15) trained 20 M longer
# with the same recipe (c5i_big improved most between 12 and 25 M).
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/clean_queue9.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
CLEAN="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-0.5 ang_acc=-0.003 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
B='dr_floor_tc=[0.02,0.8]'
EXTRA_SET="$B ref_swing_h=0.015 ref_table=\"c_big2\" gait_hz=0.3125 settle_gate_u=[0.1875,0.6875] cmd_vx=[0.01,0.03] stride_target=0.05" \
  scripts/b150_run.sh c8_big2 shift_b150 hw_ank3_b150_legs_10dof 25e6 --init runs/c5i_big/params_0025M.pkl --rw $STEPRW $CLEAN
EXTRA_SET="$B ref_swing_h=0.010 ref_table=\"c_T15\" gait_hz=0.3333333 settle_gate_u=[0.2167,0.7167] cmd_vx=[0.01,0.03] stride_target=0.04" \
  scripts/b150_run.sh c6c_T15 shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/c6_T15/params_0025M.pkl --rw $STEPRW $CLEAN
