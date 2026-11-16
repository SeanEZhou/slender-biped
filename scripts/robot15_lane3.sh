#!/bin/bash
# Round 15 lane 3: standing shoves 0.15-0.4 m/s (half of them pure sideways), capture-point steps on rs3 (the falling
# side's foot: the loaded leg steps out), random steps kept at 0.4/s, from r14f_cp (stage B on rs2).
cd "$(dirname "$0")/.."
source scripts/robot15_common.sh
bs r15b_push 16e6 runs/r14f_cp/params_0014M.pkl "rec_rand=0.4 rec_rest_s=[1.0,2.5] $CP $SHOVE push_dv=[0.15,0.4] push_lat=0.5"
echo "$(date) lane done"
