#!/bin/bash
# 2026-11-11 robot round 3, queue 4 (replaces queue 3 after its first run, h2_w6_L; only h3_k_L ran, queue 5 took over): the robot's hip rolls are ~2.5x
# stiffer than the model's (standing hip-roll sag 1.4 vs 3.5 deg; knees match) and on the robot the pelvis lifts the
# other foot 44-59 mm inside the standing board vs 21-31 mm in sim: the soft model hip sagged into the weight shift and
# did part of the policy's work. All runs train with kp_hip_roll 2.5 (`K`), on h1's recipe (c3m + Open Duck smoothness
# action_rate / action_acc -0.5, command delay 0-2 steps, `narrow`), then one twitch fix or the wide gait at a time.
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 clear=-2.0"
SMOOTH="action_rate=-0.5 action_acc=-0.5 narrow=-1.0"
IMIT="ref_q=5.0 ref_contact=5.0 upright=2.0"
T12='ref_swing_h=0.010 gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 dr_delay_steps=[0,2] kp_hip_roll=2.5 $3" scripts/b150_run.sh $1 shift_b150 $2 20e6 "${@:4}"; }
L=hw_ank3_b150_legs_10dof; F=hw_ank3_b150_10dof
W6='ref_table="c_T12w6" narrow_min=0.125 upright_sigma=0.003'
N='ref_table="c_T12" narrow_min=0.095'
run h3_k_L $L "$N" --init runs/c3m_T12/params_0020M.pkl --rw $STEPRW $C3M $SMOOTH
