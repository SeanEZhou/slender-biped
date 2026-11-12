#!/bin/bash
# 2026-11-11 robot round 3, queue 6 (after queue 5): the winners combined. Robot-like hip rolls (K, kp_hip_roll 2.5) and
# h1's smoothness recipe throughout; joint-speed noise 1.0 (h6: command jitter 0.80 -> 0.55 deg, no falls at 20 M);
# the wide gait c_T12w6 (h4: boards >= 106 mm apart, 0 falls, 65 cm / 30 s at 5 and 7.4 V); the Lipschitz penalty
# --lcp 0.0005 (h7: command jitter 0.80 -> 0.24-0.30 deg, as smooth as the filtered policies without their lag,
# 72-83 cm / 30 s, 0 falls at 10 M).
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^/bin/bash scripts/robot3_queue5.sh"; do sleep 30; done
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 clear=-2.0"
SMOOTH="action_rate=-0.5 action_acc=-0.5 narrow=-1.0"
IMIT="ref_q=5.0 ref_contact=5.0 upright=2.0 ref_swing=10.0"
T12='ref_swing_h=0.010 gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 dr_delay_steps=[0,2] kp_hip_roll=2.5 noise_qd=1.0 $3" scripts/b150_run.sh $1 shift_b150 $2 20e6 "${@:4}"; }
L=hw_ank3_b150_legs_10dof; F=hw_ank3_b150_10dof
W6='ref_table="c_T12w6" narrow_min=0.125 upright_sigma=0.01'
W3='ref_table="c_T12w3" narrow_min=0.11 upright_sigma=0.01'
N='ref_table="c_T12" narrow_min=0.095'
run h9_w6kl_L $L "$W6" --init runs/h4_w6k_L/params_0020M.pkl --lcp 0.0005 --rw $STEPRW $C3M $SMOOTH $IMIT
run h10_kl_F $F "$N" --init runs/h5_k_F/params_0010M.pkl --lcp 0.0005 --rw $STEPRW $C3M $SMOOTH
run h11_w6kl_F $F "$W6" --init runs/h8_w6k_F/params_0020M.pkl --lcp 0.0005 --rw $STEPRW $C3M $SMOOTH $IMIT
run h12_w3kl_L $L "$W3" --init runs/h7_lcp_L/params_0010M.pkl --lcp 0.0005 --rw $STEPRW $C3M $SMOOTH $IMIT
run h13_kl_L $L "$N" --init runs/h7_lcp_L/params_0010M.pkl --lcp 0.0005 --rw $STEPRW $C3M $SMOOTH
