#!/bin/bash
# 2026-11-13 round 8, queue 2 (replaces the rest of queue 1): s2 at 11M is steady (r7 0/6, r7hard 2/6, gains near h4's)
# but its steps fade, as r8's did: with the ring, standing pays (ref_contact ~2000 a walk for feet down on the
# step-settle-step schedule, ref_swing ~300 for lifting them). Pay for swinging more and for contact less.
#  s3_swing: s2_alpf_com_ring 11M -> same simulator, ref_swing 6 -> 15, ref_contact 3 -> 2
cd "$(dirname "$0")/.."
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=15.0 slip=-100.0 ref_contact=2.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.09,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
RING='ring=[["pitch",4.2,0.2,1.0,0.02],["roll",5.3,0.25,0.7,0.02]] ring_delay=0.02 dr_delay_steps=[2,3]'
STEP="progress=1.5 stride=50.0 narrow=-1.0 $BASE"
COM='dr_com_x=[-0.005,0.025]'
run s3_swing $F 24e6 "$W $LEAN $SER $RING act_lpf=0.6 $COM" --init runs/s2_alpf_com_ring/params_0011M.pkl --rw $STEP $C3M tilt_lim=-0.05
