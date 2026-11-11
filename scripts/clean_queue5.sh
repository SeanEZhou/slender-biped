#!/bin/bash
# 2026-11-10 round 5: the continuous 1.2 s walker (c3i_T12: 0 falls hard / medium rug, ~1.7 cm/s, wobble 15-20 deg/s,
# touchdown ~160-230 mm/s) fine-tuned for steadiness: body roll/pitch rate x4 and angular acceleration x3; c3r also
# trains on softer rugs (its soft-rug walks fell 4/6).
cd "$(dirname "$0")/.."
while pgrep -f "^/bin/bash scripts/clean_queue4.sh" > /dev/null; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
STEADY="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-2.0 ang_acc=-0.01 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12" \
  scripts/b150_run.sh c3s_T12 shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/c3i_T12/params_0025M.pkl --rw $STEPRW $STEADY
EXTRA_SET="dr_floor_tc=[0.1,1.2] $T12" \
  scripts/b150_run.sh c3r_T12 shift_b150 hw_ank3_b150_legs_10dof 20e6 --init runs/c3i_T12/params_0025M.pkl --rw $STEPRW $STEADY
