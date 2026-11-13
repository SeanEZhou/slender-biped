#!/bin/bash
# 2026-11-12 robot round 5, queue 2 (replaces queue 1 after fb1_s10). The simulator fitted to the 16 robot walks of
# 2026-11-12 (scripts/gap_fit.py, results/robot5): a hidden knee zero offset of -0.045 rad (the robot stands 3.7 deg back
# of the model) and a backward pull of 0.6 N per m walked, capped at 0.2 N (cable / rug), reproduce the robot's
# times to the fall (s10 23 s, s7 16, s8 18) and its backward falls. Train with them randomized around that:
#  knee offset U(-0.07, 0) common + +-0.015 per joint, hip pitch +-0.015 per joint (the legs differ), pull 0-1 N/m capped 0.3 N
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py --name fb1_s10"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" hidden_flex=[[\"hip_roll\",4.0]] $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
B=hw_ank3_b150_legs_pelvis_10dof
F=hw_ank3_b150_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_flex=[0.5,1.5] dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.07,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=1.0 drag_back_max=0.3'
RW="progress=1.0 stride=40.0 narrow=-1.0 $BASE $C3M"
run b1_lean $B 30e6 "$W $LEAN" --init runs/s10_robust/params.pkl --rw $RW
run fb2_lean $F 30e6 "$W $LEAN" --init runs/fb1_s10/params.pkl --rw $RW
run b2_lean_s8 $B 30e6 "$W $LEAN" --init runs/s8_push/params.pkl --rw progress=2.0 stride=60.0 narrow=-1.0 $BASE $C3M
run fb3_c3f $F 30e6 "$W $LEAN" --init runs/c3f_T12/params_0025M.pkl --rw $RW
