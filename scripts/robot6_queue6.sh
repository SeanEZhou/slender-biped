#!/bin/bash
# 2026-11-13 round 6, queue 6 (after h2_old of queue 5). h1_ser (mild tilt penalty) is calm but still steps in place
# episode against ~+200 for progress: every policy stopped stepping (ref_swing 773 -> ~50). It sits below the robot's
# normal walking roll. Now: past 4.6 deg (0.08 rad), -0.2, from fb2_lean (which steps).
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
C3S="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.3 action_acc=-0.3 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
L=hw_ank3_b150_legs_pelvis_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.09,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
OLDFLEX='hidden_flex=[["hip_roll",3.5]] dr_flex=[0.6,1.5]'
RW="progress=1.0 stride=40.0 narrow=-1.0 $BASE"
#  h3_notilt: fb2_lean -> re-massed + series give, NO tilt penalty, progress 1.5 / stride 50 (control: is it the penalty?)
#  l1_mass:   legs b7_seed2 -> re-massed legs model, series give, mild tilt penalty, no pull
#  h4_t05:    h3 + a very light tilt penalty (-0.05 past 4.6 deg)
STEP="progress=1.5 stride=50.0 narrow=-1.0 $BASE"
run h3_notilt $F 30e6 "$W $LEAN $SER" --init runs/fb2_lean/params.pkl --rw $STEP $C3M
run l1_mass $L 30e6 "$W $LEAN $SER" --init runs/b7_seed2/params.pkl --rw $RW $C3M tilt_lim=-0.2
run h4_t05 $F 30e6 "$W $LEAN $SER" --init runs/fb2_lean/params.pkl --rw $STEP $C3M tilt_lim=-0.05
