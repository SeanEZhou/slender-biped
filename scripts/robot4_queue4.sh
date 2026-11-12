#!/bin/bash
# 2026-11-12 robot round 4, queue 4. r2 and r3 (30 M with the give) learned to step in place (r3: touchdowns as
# c3m, forward speed 0): with the give, going forward costs falls and stride=20 alone doesn't pay for it. So: a forward
# term (progress = cycle-averaged speed / command, box average) and stride doubled; from r1_flex (already adapted to
# the give and the pelvis model).
#  s1: c3m's speeds (0.01-0.03), stride to 4 cm      s2: longer (0.02-0.04, stride to 5 cm)
#  s3: s2 on the 15 mm outer strip                    s4: s2 + mild roll_lim (-0.1 per deg^2 past 5 deg)
#  s5: s2 on the 25 mm outer strip
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
STEPRW="progress=1.0 stride=40.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
FLEX='hidden_flex=[["hip_roll",4.0]] dr_flex=[0.75,1.5] kp_hip_roll=2.5 vy_avg_mode="box"'
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $FLEX $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
I="--init runs/r1_flex/params.pkl"
B=hw_ank3_b150_legs_pelvis_10dof
LONG="$T12 cmd_vx=[0.02,0.04] stride_target=0.05"
run s1_fwd $B 20e6 "$T12 cmd_vx=[0.01,0.03] stride_target=0.04" $I --rw $STEPRW $C3M
run s2_long $B 30e6 "$LONG" $I --rw $STEPRW $C3M
run s3_w15 hw_ank3_b150_legs_pelvis_op0s0w15_10dof 30e6 "$LONG" $I --rw $STEPRW $C3M
run s4_mild $B 30e6 "$LONG roll_lim=0.087" $I --rw $STEPRW $C3M roll_lim=-0.1
run s5_w25 hw_ank3_b150_legs_pelvis_op0s0w25_10dof 30e6 "$LONG" $I --rw $STEPRW $C3M
