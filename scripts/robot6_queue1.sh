#!/bin/bash
# 2026-11-13 robot round 6 (overnight): the full body on the re-massed model (hw_ank3_b150_m2_10dof: legs 0.9 kg, upper
# body 0.3 kg, docs/measuring.md "Full body"), no cable pull, a penalty on tilting past ~3.4 deg (roll and backward pitch).
#  g1_mass:   fb2_lean on the re-massed model, round-5 give (hip roll k 3.5 quasi-static), tilt penalty
#  g2_series: g1 + the give as a series spring at hip roll / ankle roll (k 10, soft k2 2.5 past 0.15 N m), knee and hip
#             pitch (k 15), hip yaw (k 10), randomized x0.5-1.5, time constant 3-8 ms (E7 taps: the body rings)
#  g3_tilt:   g2 + a stronger, earlier tilt penalty (2.9 deg, -3)
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.09,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
OLDFLEX='hidden_flex=[["hip_roll",3.5]] dr_flex=[0.6,1.5]'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
RW="progress=1.0 stride=40.0 narrow=-1.0 $BASE $C3M"
run g1_mass $F 30e6 "$W $LEAN $OLDFLEX" --init runs/fb2_lean/params.pkl --rw $RW tilt_lim=-1.0
run g2_series $F 30e6 "$W $LEAN $SER" --init runs/g1_mass/params.pkl --rw $RW tilt_lim=-1.0
run g3_tilt $F 30e6 "$W $LEAN $SER tilt_lim=0.05" --init runs/g2_series/params.pkl --rw $RW tilt_lim=-3.0
