#!/bin/bash
# 2026-11-13 round 7, queue 4 (after queue 3). r1_raw / r3_hist still chatter in the ringing sim (4-8 Hz commands
# 4.6-7%, 4/9 falls on r7). On the robot the action low-pass 0.6 helped most (h4: 42 s, 65 steps): train with it.
#  r8_alpf_h4:   h4_t05 -> ringing sim, act_lpf 0.6 (the robot runner applies the run's act_lpf by itself)
#  r9_alpf_hist: r3_hist -> ringing sim, act_lpf 0.6, history
#  r10_lcp_h4:   h4_t05 -> ringing sim, Lipschitz penalty 0.0005 on the sensor inputs (lower gain round the loop)
cd "$(dirname "$0")/.."
while kill -0 1380005 2>/dev/null; do sleep 30; done
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
BASE="ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
F=hw_ank3_b150_m2_10dof
W="$T12 cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN7='hidden_off=[["knee",-0.05,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
RING='ring=[["pitch",4.2,0.2,1.0,0.02],["roll",5.3,0.25,0.7,0.02]] ring_delay=0.02 dr_delay_steps=[2,3]'
RW="progress=1.0 stride=40.0 narrow=-1.0 $BASE"
S7="$W $LEAN7 $SER $RING"
run r8_alpf_h4 $F 30e6 "$S7 act_lpf=0.6" --init runs/h4_t05/params.pkl --rw $RW $C3M tilt_lim=-0.05
run r9_alpf_hist $F 30e6 "$S7 obs_hist=5 act_lpf=0.6" --init runs/r3_hist/params.pkl --rw $RW $C3M tilt_lim=-0.05
run r10_lcp_h4 $F 30e6 "$S7" --init runs/h4_t05/params.pkl --lcp 0.0005 --rw $RW $C3M tilt_lim=-0.05
