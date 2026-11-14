#!/bin/bash
# 2026-11-13 round 7, queue 2 (after queue 1):
#  r4_smooth:  r1's recipe + 3x smoothness (action_rate / action_acc -0.3): the standing twitch
#  r5_histlpf: history + gyro low-pass
#  r6_hard:    raw gyro, trained around the harsh ring (eval6 --sim r7hard: gain 1.4, zeta 0.15, knee -0.045)
cd "$(dirname "$0")/.."
while kill -0 1362494 2>/dev/null; do sleep 30; done
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
C3S="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.3 action_acc=-0.3 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN7='hidden_off=[["knee",-0.05,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
RING='ring=[["pitch",4.2,0.2,1.0,0.02],["roll",5.3,0.25,0.7,0.02]] ring_delay=0.02 dr_delay_steps=[2,3]'
RINGH='ring=[["pitch",4.2,0.15,1.4,0.02],["roll",5.3,0.18,1.0,0.02]] ring_delay=0.025 dr_delay_steps=[2,3]'
RW="progress=1.0 stride=40.0 narrow=-1.0 $BASE"
S7="$W $LEAN7 $SER $RING"
run r4_smooth $F 30e6 "$S7" --init runs/b7_seed2/params.pkl --rw $RW $C3S tilt_lim=-0.05
run r5_histlpf $F 30e6 "$S7 obs_hist=5 gyro_lpf=2.5" --init runs/c0_hist/params.pkl --rw $RW $C3M tilt_lim=-0.05
run r6_hard $F 30e6 "$W $LEAN7 $SER $RINGH" --init runs/b7_seed2/params.pkl --rw $RW $C3M tilt_lim=-0.05
