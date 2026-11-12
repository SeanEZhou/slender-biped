#!/bin/bash
# 2026-11-11 robot round 3, queue 5 (after h3_k_L). All with the robot-like hip rolls (kp_hip_roll 2.5, `K`) on h1's
# recipe (c3m + Open Duck smoothness, command delay 0-2 steps, `narrow`).
# h2_w6_L (wide gait from c3m, 20 M): boards 158 mm apart (min 117) but it stopped walking (8 cm / 30 s; 10 M: 24 cm):
# at the switch the swing lift collapsed (ref_swing 941 -> 142) and the tight upright kernel fell 712 -> 382 late.
# Wide runs now continue from h2_w6_L 10 M with a looser upright kernel (0.01), ref_swing 10 (was 6).
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 clear=-2.0"
SMOOTH="action_rate=-0.5 action_acc=-0.5 narrow=-1.0"
IMIT="ref_q=5.0 ref_contact=5.0 upright=2.0 ref_swing=10.0"
T12='ref_swing_h=0.010 gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 dr_delay_steps=[0,2] kp_hip_roll=2.5 $3" scripts/b150_run.sh $1 shift_b150 $2 20e6 "${@:4}"; }
L=hw_ank3_b150_legs_10dof; F=hw_ank3_b150_10dof
W6='ref_table="c_T12w6" narrow_min=0.125 upright_sigma=0.01'
N='ref_table="c_T12" narrow_min=0.095'
run h4_w6k_L $L "$W6" --init runs/h2_w6_L/params_0010M.pkl --rw $STEPRW $C3M $SMOOTH $IMIT
run h5_k_F $F "$N" --init runs/c3f_T12/params_0025M.pkl --rw $STEPRW $C3M $SMOOTH
run h6_nq_L $L "$N noise_qd=1.0" --init runs/c3m_T12/params_0020M.pkl --rw $STEPRW $C3M $SMOOTH
run h7_lcp_L $L "$N" --init runs/c3m_T12/params_0020M.pkl --lcp 0.0005 --rw $STEPRW $C3M $SMOOTH
run h8_w6k_F $F "$W6" --init runs/h4_w6k_L/params_0020M.pkl --rw $STEPRW $C3M $SMOOTH $IMIT
