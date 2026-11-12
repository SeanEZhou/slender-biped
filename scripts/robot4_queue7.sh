#!/bin/bash
# 2026-11-12 robot round 4, queue 7 (after queue 6): margin for a softer real hip.
#  s10: s7 + give k 2-6 (dr_flex 0.5-1.5) + IMU tilt offsets up to 3 deg (the robot stands 3.7 deg back), 30 M
#  s11: s8 (longer steps) + narrow x3 and give k 2-6, to win back s7's margin at k = 3
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^/bin/bash scripts/robot4_queue6.sh"; do sleep 30; done
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" hidden_flex=[[\"hip_roll\",4.0]] $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
B=hw_ank3_b150_legs_pelvis_10dof
W="$T12 dr_flex=[0.5,1.5] cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065"
run s10_robust $B 30e6 "$W dr_imu_tilt=0.05" --init runs/s7_narrow/params.pkl --rw progress=1.0 stride=40.0 narrow=-1.0 $BASE $C3M
run s11_pushn $B 30e6 "$W" --init runs/s8_push/params.pkl --rw progress=2.0 stride=60.0 narrow=-3.0 $BASE $C3M
