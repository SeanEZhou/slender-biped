#!/bin/bash
# 2026-11-12 robot round 4, queue 5 (after queue 4). s2_long: 0/12 falls at every speed at the calibrated give (k=4),
# but 4/12 at k=3 and 0.015 (outside its 0.02-0.04 training range). Fine-tune s2 for margin:
#  s6: give randomized wider (k 2.4-6), commands 0.015-0.04
#  s7: s6 + narrow only (-1 below 65 mm; s4 showed even a mild roll_lim hurts)
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^/bin/bash scripts/robot4_queue4.sh"; do sleep 30; done
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
STEPRW="progress=1.0 stride=40.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
FLEX='hidden_flex=[["hip_roll",4.0]] dr_flex=[0.6,1.5] kp_hip_roll=2.5 vy_avg_mode="box"'
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $FLEX $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
I="--init runs/s2_long/params.pkl"
B=hw_ank3_b150_legs_pelvis_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05"
run s6_wide $B 30e6 "$W" $I --rw $STEPRW $C3M
run s7_narrow $B 30e6 "$W narrow_min=0.065" $I --rw $STEPRW $C3M narrow=-1.0
