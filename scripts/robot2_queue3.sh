#!/bin/bash
# 2026-11-11 round "robot feedback 3", after queue 2: stride 40 (e1_str_F) is the only change that lengthened the steps
# (42 cm / 30 s on 5 V, shake 5.1 deg/s) but its boards touch now and then; ref_q 6 and track_lin 3 shortened the
# steps (the latter shuffles); gentle ramped pulls (e4) taught stepping in place. Strong sudden pulls up to 1.5 N
# (l2_tug15_L, legs) were the only pull training that worked: 0/6 falls at 0.5 N ramped, against 5/6 for c3m.
#   TUG15 sudden pulls 0-1.5 N, half the time (as l2)
cd "$(dirname "$0")/.."
while ps -eo pid,args | grep -q "^ *[0-9]* /bin/bash scripts/robot2_queue2.sh"; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
MID="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.3 action_acc=-0.1 clear=-2.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
BASE='act_lpf=0.7 dr_prof_acc=[0.6,4.0] mirror_aug=true cmd_vx=[0.02,0.04] stride_target=0.05'
TUG15='tug_force=1.5'
F=hw_ank3_b150_10dof; L=hw_ank3_b150_legs_10dof
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 $BASE $1" scripts/b150_run.sh $2 shift_b150 $3 20e6 --init $4 --rw $STEPRW $MID $5; }
run "$TUG15" f1_tug15_F  $F runs/d4_lpf7_F/params_0020M.pkl ""
run "$TUG15" f2_strtug_F $F runs/e1_str_F/params_0020M.pkl "stride=40.0"
run ""       f3_stry_F   $F runs/e1_str_F/params_0020M.pkl "stride=40.0 board_yaw=-100.0"
run "$TUG15" f4_strtug_L $L runs/l2_tug15_L/params_0020M.pkl "stride=40.0"
run "$TUG15" f5_tug15_L  $L runs/l2_tug15_L/params_0020M.pkl ""
