#!/bin/bash
# 2026-11-11 robot round 3: servo twitch (the policy's own command jitter, 0.45-0.6 deg per 100 ms on the robot) and
# feet drifting together. h1: c3m's recipe with Open Duck's smoothness (action_rate -0.5, action_acc -0.5 vs -0.1),
# command delay 0-2 steps (Open Duck 0-3), and a penalty on boards closer than 95 mm (nominal 101 mm). No action filter.
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 clear=-2.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
SMOOTH="action_rate=-0.5 action_acc=-0.5 narrow=-1.0"
EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 dr_delay_steps=[0,2] narrow_min=0.095" scripts/b150_run.sh h1_od_L shift_b150 hw_ank3_b150_legs_10dof 20e6 \
  --init runs/c3m_T12/params_0020M.pkl --rw $STEPRW $C3M $SMOOTH
