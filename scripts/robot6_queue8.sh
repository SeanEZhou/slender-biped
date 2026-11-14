#!/bin/bash
# 2026-11-13 round 6, queue 8 (last): h7_h5step: h5_h2hard (the most robust stepping policy, half h2's progress) with
# more forward pay (progress 1.5, stride 50), same hard margins and mild tilt penalty.
cd "$(dirname "$0")/.."
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN2='hidden_off=[["knee",-0.13,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
run h7_h5step $F 30e6 "$W $LEAN2 hidden_flex=[[\"hip_roll\",3.5]] dr_flex=[0.4,1.5]" --init runs/h5_h2hard/params.pkl --rw progress=1.5 stride=50.0 narrow=-1.0 $BASE $C3M tilt_lim=-0.2
