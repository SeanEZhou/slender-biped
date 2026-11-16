#!/bin/bash
# Round 15 lane 2: walk, recover, resume. k5 (k5_high2 at 16M) on its own recipe, the rs3 recovery steps between
# (rec_walk): shoves while walking and standing; the capture point past 0.08 m while walking (0.035 standing) starts a
# step, random steps at 0.2/s walking and 0.5/s standing; after a step the gait clock restarts and the forward command
# ramps up from 0. The command inputs keep k5's weights, rescaled (--reset-cmd-keep): k5 walks as before.
cd "$(dirname "$0")/.."
source scripts/robot15_common.sh
name=r15w_walk
[ -f runs/$name/done.json ] && exit 0
echo "$(date) start $name"
run $name $F 8e6 "$T12 $W $LEAN $SER $HIGH2 rec_table=\"rs3\" rec_walk=true cmd_zero_prob=0.2 rec_rand=0.5 rec_rand_walk=0.2 rec_rest_s=[1.0,2.5] $CP rec_cp_th_walk=0.08 push_dv=[0.1,0.35] push_dur=[0.08,0.2] push_interval_s=[3.0,6.0] push_walk=true push_lat=0.3" \
    --init runs/k5_high2/params_0016M.pkl --reset-cmd 0.04,0.03,0.5 --reset-cmd-keep \
    --rw $STEP $C3M tilt_lim=-0.05 ref_q=6.0 ref_contact=6.0 ref_swing=6.0
echo "$(date) end $name rc=$?"
