#!/bin/bash
# 2026-11-10 round 12 (last of the session): the 1.6 s big-step hybrid (c8_big2: 0/16 falls, 38-45 mm lift, boards
# touching in 7/16 walks) with the middle steadiness weights and the `narrow` term.
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/clean_queue11.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
MID="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BIG2='ref_swing_h=0.015 ref_table="c_big2" gait_hz=0.3125 settle_gate_u=[0.1875,0.6875] cmd_vx=[0.01,0.03] stride_target=0.05'
EXTRA_SET="dr_floor_tc=[0.02,0.8] $BIG2" \
  scripts/b150_run.sh c8m_big2 shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/c8_big2/params_0025M.pkl --rw $STEPRW $MID narrow=-1.0
