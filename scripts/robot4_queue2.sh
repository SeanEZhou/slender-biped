#!/bin/bash
# 2026-11-12 robot round 4, queue 2 (after queue 1): the same recipe as r3_long (hidden hip give, roll_lim, narrow,
# longer steps) on boards widened outward by the clip-on outer strip (hardware/feet_p1/cad/outer_strip.py).
#  p1: 15 mm strip, p2: 25 mm strip; both from c3m_T12.
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "^/bin/bash scripts/robot4_queue1.sh"; do sleep 30; done
while ps -eo args | grep -q "^scripts/../.venv/bin/python scripts/train.py"; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
LIM="roll_lim=-1.0 narrow=-1.0"
FLEX='hidden_flex=[["hip_roll",4.0]] dr_flex=[0.75,1.5] kp_hip_roll=2.5'
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $FLEX $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
I="--init runs/c3m_T12/params_0020M.pkl"
LONG="$T12 cmd_vx=[0.02,0.04] stride_target=0.05 roll_lim=0.07"
run p1_w15 hw_ank3_b150_legs_pelvis_op0s0w15_10dof 30e6 "$LONG narrow_min=0.085" $I --rw $STEPRW $C3M $LIM
run p2_w25 hw_ank3_b150_legs_pelvis_op0s0w25_10dof 30e6 "$LONG narrow_min=0.095" $I --rw $STEPRW $C3M $LIM
