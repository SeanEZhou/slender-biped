#!/bin/bash
# 2026-11-12 robot round 4, queue 6 (after queue 5): longer steps. s2 makes ~25 mm steps at 0.04 where the c_T12
# reference asks for 48 mm; push the forward terms harder.
#  s8: from s7_narrow (best: 0/12 at k=3-4, all speeds), progress=2, stride=60 (to 5 cm), narrow kept
#  s9: the s2 recipe on the slower c_T15 gait (1.5 s a step), from r1_flex
cd "$(dirname "$0")/.."

while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
T15='ref_swing_h=0.010 ref_table="c_T15" gait_hz=0.3333333 settle_gate_u=[0.2167,0.7167]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" hidden_flex=[[\"hip_roll\",4.0]] $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
B=hw_ank3_b150_legs_pelvis_10dof
run s8_push $B 30e6 "$T12 dr_flex=[0.6,1.5] cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065" --init runs/s7_narrow/params.pkl --rw progress=2.0 stride=60.0 narrow=-1.0 $BASE $C3M
run s9_t15 $B 30e6 "$T15 dr_flex=[0.6,1.5] cmd_vx=[0.0175,0.035] stride_target=0.05 narrow_min=0.065" --init runs/s7_narrow/params.pkl --rw progress=1.0 stride=40.0 narrow=-1.0 $BASE $C3M
