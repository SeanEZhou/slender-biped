#!/bin/bash
# 2026-11-10 evening: full body (torso + head on, models/hw_ank3_b150_10dof, 1.45 kg) from c3m_T12, the policy that
# walked best on the robot (legs only; c6_T15 stepped on itself, c5i_big tipped every other step). Same recipe as c3m.
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
MID="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12" \
  scripts/b150_run.sh c3f_T12 shift_b150 hw_ank3_b150_10dof 25e6 --init runs/c3m_T12/params_0020M.pkl --rw $STEPRW $MID
