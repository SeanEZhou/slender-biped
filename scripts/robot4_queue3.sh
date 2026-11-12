#!/bin/bash
# 2026-11-12 robot round 4, queue 3 (replaces the rest of queues 1-2): r2_lim (roll_lim -1 at 4 deg + narrow) stopped
# walking (0/12 falls but ~0 cm forward), so the limits go in mildly or not at all.
#  r3: longer steps (commands 0.02-0.04, stride to 5 cm), give, no limit terms
#  r5: r3 + mild roll_lim (-0.1 per deg^2 past 5 deg) + narrow
#  p1: r3 on the 15 mm outer strip;  r4: r3 on the slower c_T15 gait;  p2: r3 on the 25 mm strip
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
FLEX='hidden_flex=[["hip_roll",4.0]] dr_flex=[0.75,1.5] kp_hip_roll=2.5'
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
T15='ref_swing_h=0.010 ref_table="c_T15" gait_hz=0.3333333 settle_gate_u=[0.2167,0.7167]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $FLEX $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
I="--init runs/c3m_T12/params_0020M.pkl"
B=hw_ank3_b150_legs_pelvis_10dof
run r3_long $B 30e6 "$T12 cmd_vx=[0.02,0.04] stride_target=0.05" $I --rw $STEPRW $C3M
run r5_mild $B 30e6 "$T12 cmd_vx=[0.02,0.04] stride_target=0.05 roll_lim=0.087 narrow_min=0.07" $I --rw $STEPRW $C3M roll_lim=-0.1 narrow=-1.0
run p1_w15 hw_ank3_b150_legs_pelvis_op0s0w15_10dof 30e6 "$T12 cmd_vx=[0.02,0.04] stride_target=0.05" $I --rw $STEPRW $C3M
run r4_t15 $B 30e6 "$T15 cmd_vx=[0.0175,0.035] stride_target=0.05" $I --rw $STEPRW $C3M
run p2_w25 hw_ank3_b150_legs_pelvis_op0s0w25_10dof 30e6 "$T12 cmd_vx=[0.02,0.04] stride_target=0.05" $I --rw $STEPRW $C3M
