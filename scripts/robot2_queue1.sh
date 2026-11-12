#!/bin/bash
# 2026-11-11 round "robot feedback 1". On the robot: c3f_T12 (full body) shakes sideways at ~4.8 Hz (its hip / ankle
# roll commands oscillate in phase with the roll rate; worse at 7.4 V), marches nearly in place with the right foot
# stepping less than the left; c3m_T12 (legs only) is pulled over by tense cables. Fixes tried, one at a time and
# together, all fine-tuned 20 M steps from the policy the robot ran:
#   LPF  action low-pass (act_lpf) and a heavier action-rate penalty: less 5 Hz content in the commands
#   FAST servo profiles randomized up to 4x faster (dr_prof_acc): the real servos pass ~2x more 5 Hz motion than the
#        fitted model, so the simulated loop never shook
#   PROG mirrored episodes (left/right symmetric gait) and a higher speed command / stride target
#   TUG  a steady random pull on the pelvis (a tense cable)
cd "$(dirname "$0")/.."
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
MID="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
FAST='dr_prof_acc=[0.6,4.0]'
PROG='mirror_aug=true cmd_vx=[0.02,0.04] stride_target=0.05'
F=hw_ank3_b150_10dof; L=hw_ank3_b150_legs_10dof
INF=runs/c3f_T12/params_0025M.pkl; INL=runs/c3m_T12/params_0020M.pkl
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 $1" scripts/b150_run.sh $2 shift_b150 $3 20e6 --init $4 --rw $STEPRW $MID $5; }
run "act_lpf=0.5 $FAST"             d1_lpf_F  $F $INF "action_rate=-0.3"
run "$PROG"                         d2_sym_F  $F $INF ""
run "act_lpf=0.5 $FAST $PROG"       d3_both_F $F $INF "action_rate=-0.3"
run "act_lpf=0.7 $FAST $PROG"       d4_lpf7_F $F $INF "action_rate=-0.3"
run "act_lpf=0.5 $FAST $PROG tug_force=1.0" d5_tug_F $F $INF "action_rate=-0.3"
run "act_lpf=0.5 $FAST $PROG tug_force=1.0" l1_tug_L $L $INL "action_rate=-0.3"
run "act_lpf=0.5 $FAST tug_force=1.5"       l2_tug15_L $L $INL "action_rate=-0.3"
