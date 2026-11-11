#!/bin/bash
# 2026-11-10 round 9: the calmer 1.5 s continuous walker (c6_T15 at 25 M: 0/16 falls hard + medium rug, touchdown
# ~110 mm/s, 4.8 deg / 39 deg/s after landing, 26-32 mm lift) with the middle steadiness weights (x2).
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/clean_queue8.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
MID="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
T15='ref_swing_h=0.010 ref_table="c_T15" gait_hz=0.3333333 settle_gate_u=[0.2167,0.7167] cmd_vx=[0.01,0.03] stride_target=0.04'
EXTRA_SET="dr_floor_tc=[0.02,0.8] $T15" \
  scripts/b150_run.sh c6m_T15 shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/c6_T15/params_0025M.pkl --rw $STEPRW $MID
