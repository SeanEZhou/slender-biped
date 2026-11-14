#!/bin/bash
# 2026-11-13 round 6, queue 7 (after queue 6). h2_old walks (fitted 0/9, 32 cm) but the hard sim breaks it (7/9):
#  h5_h2hard: h2_old + harder training margins (knee lean to -0.13, give x0.4-1.5), same mild tilt penalty
#  h6_h2s1:   h2_old's recipe from fb2_lean, training seed 1 (does the recipe repeat?)
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^/bin/bash scripts/robot6_queue6.sh"; do sleep 30; done
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.09,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
LEAN2='hidden_off=[["knee",-0.13,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
RW="progress=1.0 stride=40.0 narrow=-1.0 $BASE"
run h5_h2hard $F 30e6 "$W $LEAN2 hidden_flex=[[\"hip_roll\",3.5]] dr_flex=[0.4,1.5]" --init runs/h2_old/params.pkl --rw $RW $C3M tilt_lim=-0.2
run h6_h2s1 $F 30e6 "$W $LEAN hidden_flex=[[\"hip_roll\",3.5]] dr_flex=[0.6,1.5]" --init runs/fb2_lean/params.pkl --seed 1 --rw $RW $C3M tilt_lim=-0.2
