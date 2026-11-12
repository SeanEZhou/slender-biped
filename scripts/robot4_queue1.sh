#!/bin/bash
# 2026-11-12 robot round 4, queue 1: train on the model with the pelvis shell and the encoder-invisible hip give
# calibrated on the E1 tilt test (hip_roll k = 4 N*m/rad, randomized 3-6), from c3m_T12 (the robot's best walker).
#  r1: c3m's recipe, give on (adaptation only)
#  r2: + roll_lim (pelvis roll past 4 deg, where the real feet slip inward) + narrow (keep the board gap >= 70 mm)
#  r3: r2 + longer steps: commands 0.02-0.04 m/s (2.4-4.8 cm a step), stride rewarded up to 5 cm
#  r4: r3 on the slower c_T15 gait (1.5 s a step: more time per step for a longer one)
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
LIM="roll_lim=-1.0 narrow=-1.0"
FLEX='hidden_flex=[["hip_roll",4.0]] dr_flex=[0.75,1.5] kp_hip_roll=2.5'
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
T15='ref_swing_h=0.010 ref_table="c_T15" gait_hz=0.3333333 settle_gate_u=[0.2167,0.7167]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $FLEX $3" scripts/b150_run.sh $1 shift_b150 hw_ank3_b150_legs_pelvis_10dof $2 "${@:4}"; }
I="--init runs/c3m_T12/params_0020M.pkl"
run r1_flex 20e6 "$T12 cmd_vx=[0.01,0.03] stride_target=0.04" $I --rw $STEPRW $C3M
run r2_lim 20e6 "$T12 cmd_vx=[0.01,0.03] stride_target=0.04 roll_lim=0.07 narrow_min=0.07" $I --rw $STEPRW $C3M $LIM
run r3_long 30e6 "$T12 cmd_vx=[0.02,0.04] stride_target=0.05 roll_lim=0.07 narrow_min=0.07" $I --rw $STEPRW $C3M $LIM
run r4_t15 30e6 "$T15 cmd_vx=[0.0175,0.035] stride_target=0.05 roll_lim=0.07 narrow_min=0.07" $I --rw $STEPRW $C3M $LIM
