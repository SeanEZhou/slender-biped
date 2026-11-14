#!/bin/bash
# 2026-11-13 round 6, queue 3 (after queue 2). g1_mass stepped in place under the tilt penalty (8 cm / 30 s):
#  g6_step:  g2_series recipe from g1_mass (g2 itself stalled: -0.3 cm, falls forward) + more forward pay (progress 1.5, stride 50) with the tilt penalty
#  g7_step3: g6 recipe with the stronger tilt penalty of g3 (2.9 deg, -3)
#  l1_mass:  legs b7_seed2 on the re-massed legs model (0.9 kg) with the series give and the tilt penalty, no pull
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^/bin/bash scripts/robot6_queue2.sh"; do sleep 30; done
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
L=hw_ank3_b150_legs_pelvis_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.09,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
STEP="progress=1.5 stride=50.0 narrow=-1.0 $BASE $C3M"
run g6_step $F 30e6 "$W $LEAN $SER" --init runs/g1_mass/params.pkl --rw $STEP tilt_lim=-1.0
run g7_step3 $F 30e6 "$W $LEAN $SER tilt_lim=0.05" --init runs/g6_step/params.pkl --rw $STEP tilt_lim=-3.0
run l1_mass $L 30e6 "$W $LEAN $SER" --init runs/b7_seed2/params.pkl --rw progress=1.0 stride=40.0 narrow=-1.0 $BASE $C3M tilt_lim=-1.0
#  g8_oldstep: g1_mass (round-5 quasi-static give, re-massed) + forward pay, lighter tilt penalty
OLDFLEX='hidden_flex=[["hip_roll",3.5]] dr_flex=[0.6,1.5]'
run g8_oldstep $F 30e6 "$W $LEAN $OLDFLEX" --init runs/g1_mass/params.pkl --rw $STEP tilt_lim=-0.5
