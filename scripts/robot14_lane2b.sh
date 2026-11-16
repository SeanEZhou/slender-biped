#!/bin/bash
# Round 14 stage B (queued 13:58): once r14c2_slow finishes, shoves 0.15-0.4 m/s at full size (no ramp), capture-point
# steps on rs2, random steps kept at 0.4/s, from r14c2_slow at 24M. Short (14M) to end by ~15:05.
cd "$(dirname "$0")/.."
source scripts/robot14_common.sh
while [ ! -f runs/r14c2_slow/done.json ]; do sleep 20; done
REC='rec_table="rs2" cmd_zero_prob=1.0 heading_cmd=false settle_gate_u=[]'
INIT=runs/r14c2_slow/params_0024M.pkl
br r14f_cp 14e6 "rec_rand=0.4 rec_rest_s=[1.0,2.5] $CP $SHOVE push_dv=[0.15,0.4]"
echo "$(date) lane done"
