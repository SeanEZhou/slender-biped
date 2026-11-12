#!/bin/bash
# 2026-11-11 last slot: e6_tugr_L (legs, furthest walker of the night, 55-64 cm / 30 s) with the strong sudden pulls
# (0-1.5 N) that gave l2 / f1 / f4 their pull tolerance.
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
MID="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.3 action_acc=-0.1 clear=-2.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
BASE='act_lpf=0.7 dr_prof_acc=[0.6,4.0] mirror_aug=true cmd_vx=[0.02,0.04] stride_target=0.05'
EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 $BASE tug_force=1.5" scripts/b150_run.sh g1_fartug_L shift_b150 hw_ank3_b150_legs_10dof 20e6 \
  --init runs/e6_tugr_L/params_0020M.pkl --rw $STEPRW $MID
