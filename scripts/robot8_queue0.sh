#!/bin/bash
# 2026-11-13 round 8 (1 h of GPU): h4_t05 with the action filter trained in, on h4's own round-6 simulator, plus a
# forward-shifted centre of mass. On the robot the round-7 policies walk as the simulator does with the pelvis mass
# 1-2 cm forward (scripts/robot8_analysis/lean.py); h4 resists that best, the filtered / LCP policies sag onto their toes.
#  s1_alpf_com: h4_t05 -> round-6 sim, act_lpf 0.6 (as on the robot), pelvis centre of mass +[-0.5, 2.5] cm
cd "$(dirname "$0")/.."
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.09,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
STEP="progress=1.5 stride=50.0 narrow=-1.0 $BASE"
COM='dr_com_x=[-0.005,0.025]'
run s1_alpf_com $F 25e6 "$W $LEAN $SER act_lpf=0.6 $COM" --init runs/h4_t05/params.pkl --rw $STEP $C3M tilt_lim=-0.05
