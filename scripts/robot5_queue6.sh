#!/bin/bash
# 2026-11-12 robot round 5, queue 6: b7_seed2 passes the fitted sim (1/24) with 27-29 mm steps; b6_mid (b1 + progress
# 1.5, stride 50) was steadier than b1. The same step from b7.
cd "$(dirname "$0")/.."
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" hidden_flex=[[\"hip_roll\",4.0]] $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
B=hw_ank3_b150_legs_pelvis_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_flex=[0.5,1.5] dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.07,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=1.0 drag_back_max=0.3'
run b8_mid2 $B 30e6 "$W $LEAN" --init runs/b7_seed2/params.pkl --rw progress=1.5 stride=50.0 narrow=-1.0 $BASE $C3M
