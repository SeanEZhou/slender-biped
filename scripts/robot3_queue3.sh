#!/bin/bash
# 2026-11-11 robot round 3, queue 3 (only its first run, h2_w6_L, ran: queue 4 replaced the rest). Queue 2's first run (h2: --lcp 0.002 + speed noise 1.0 + stiffness 0.75-2.5x)
# collapsed (episode length 178 at 10 M; lcp_grad2 36 -> 5.6): the fixes are now tried one at a time on h1's recipe.
#  h1 recipe = c3m + Open Duck smoothness (action_rate -0.5, action_acc -0.5) + command delay 0-2 steps + `narrow`
#  * wide gaits c_T12w6 / c_T12w10 (hips abducted 0.06 / 0.10 rad, ankles keep boards flat; boards 138 / 160 mm apart
#    at the ready pose vs 106), `narrow` 10 mm under that, ref_q 5 / ref_contact 5 (were 3), torso imitation (upright 2
#    with a 0.003 kernel: the reference keeps the pelvis level)
#  * joint-speed noise 1.0 rad/s (was 0.3; Open Duck 2.5) + servo stiffness 0.75-2.5x (robot hip rolls sag less)
#  * Lipschitz penalty (--lcp 0.0005, a quarter of h2's)
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 clear=-2.0"
SMOOTH="action_rate=-0.5 action_acc=-0.5 narrow=-1.0"
IMIT="ref_q=5.0 ref_contact=5.0 upright=2.0"
T12='ref_swing_h=0.010 gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 dr_delay_steps=[0,2] $3" scripts/b150_run.sh $1 shift_b150 $2 20e6 "${@:4}"; }
L=hw_ank3_b150_legs_10dof; F=hw_ank3_b150_10dof
W6='ref_table="c_T12w6" narrow_min=0.125 upright_sigma=0.003'
N='ref_table="c_T12" narrow_min=0.095'
run h2_w6_L $L "$W6" --init runs/c3m_T12/params_0020M.pkl --rw $STEPRW $C3M $SMOOTH $IMIT
