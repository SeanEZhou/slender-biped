#!/bin/bash
# 2026-11-11 round "robot feedback 2", after queue 1: the action filter (act_lpf 0.7, d4_lpf7_F) is the calmest
# (shake at 7.4 V below c3f's at 5 V, 0 falls) but walks less far; mirroring evened the feet; a higher speed command
# alone didn't lengthen the steps (policies take ~half the reference's step); sudden random pulls (d5) taught nothing:
# 0.3 N ramped is fine, 0.5 N fells half the walks with or without that training.
#   STR  longer steps: stride 40 (was 20), stride_target 0.05
#   REF  follow the reference's steps more: ref_q 6 (was 3)
#   TRK  pay for the commanded speed more: track_lin 3 (was 1)
#   TUGR pulls that build over 1 s (a cable tightening), 0-0.6 N, 70 % of the time
cd "$(dirname "$0")/.."
while ps -eo pid,args | grep -q "^ *[0-9]* /bin/bash scripts/robot2_queue1.sh"; do sleep 30; done
STEPRW="progress=0.0 stride=20.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
MID="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.3 action_acc=-0.1 clear=-2.0"
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] cmd_vx=[0.01,0.03] stride_target=0.04'
BASE='act_lpf=0.7 dr_prof_acc=[0.6,4.0] mirror_aug=true cmd_vx=[0.02,0.04] stride_target=0.05'
TUGR='tug_force=0.6 tug_prob=0.7 tug_ramp_s=1.0 tug_s=[2.0,6.0]'
F=hw_ank3_b150_10dof; L=hw_ank3_b150_legs_10dof
IND=runs/d4_lpf7_F/params_0020M.pkl; INL=runs/c3m_T12/params_0020M.pkl
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] $T12 $BASE $1" scripts/b150_run.sh $2 shift_b150 $3 20e6 --init $4 --rw $STEPRW $MID $5; }
run ""      e1_str_F  $F $IND "stride=40.0"
run ""      e2_ref6_F $F $IND "ref_q=6.0"
run ""      e3_trk_F  $F $IND "track_lin=3.0"
run "$TUGR" e4_tugr_F $F $IND ""
run ""      e5_calm_L $L $INL ""
run "$TUGR" e6_tugr_L $L $INL ""
