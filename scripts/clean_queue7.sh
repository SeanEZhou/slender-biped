#!/bin/bash
# 2026-11-10 round 7: the big step (c5i_big at 25 M: 0 falls hard / medium rug, ~55 mm lift, touchdown ~90 mm/s,
# post-landing rate ~32 deg/s, the calmest landings so far) continued with the steadiness weights (round 5).
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/clean_queue6.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
STEADY="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-2.0 ang_acc=-0.01 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BIG='ref_swing_h=0.015 ref_table="c_big" gait_hz=0.25 settle_gate_u=[0.225,0.725] cmd_vx=[0.01,0.03] stride_target=0.06'
EXTRA_SET="dr_floor_tc=[0.02,0.8] $BIG" \
  scripts/b150_run.sh c5s_big shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/c5i_big/params_0025M.pkl --rw $STEPRW $STEADY
