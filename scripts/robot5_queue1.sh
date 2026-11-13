#!/bin/bash
# 2026-11-12 robot round 5, queue 1: the round-4 recipe (hip give k = 4 randomized, progress + stride, narrow) on the
# full body (models/hw_ank3_b150_10dof: torso and head, 1.45 kg). c3f_T12 balances on the robot but steps in place.
#  fb1: from s10_robust (the steadiest legs policy on the robot), s10's recipe
#  fb2: from s7_narrow, s7's recipe (give k 2.4-6, no IMU offsets)
#  fb3: from c3f_T12 (already full body), s10's recipe
cd "$(dirname "$0")/.."
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" hidden_flex=[[\"hip_roll\",4.0]] $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065"
run fb1_s10 $F 30e6 "$W dr_flex=[0.5,1.5] dr_imu_tilt=0.05" --init runs/s10_robust/params.pkl --rw progress=1.0 stride=40.0 narrow=-1.0 $BASE $C3M
run fb2_s7 $F 30e6 "$W dr_flex=[0.6,1.5]" --init runs/s7_narrow/params.pkl --rw progress=1.0 stride=40.0 narrow=-1.0 $BASE $C3M
run fb3_c3f $F 30e6 "$W dr_flex=[0.5,1.5] dr_imu_tilt=0.05" --init runs/c3f_T12/params_0025M.pkl --rw progress=1.0 stride=40.0 narrow=-1.0 $BASE $C3M
