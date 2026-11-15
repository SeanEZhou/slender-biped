#!/bin/bash
# Round 13 queue 3 (runs alongside queue 1/2 on the same GPU): stepping shaping. r13a (stand + shoves, k5's rewards)
# resisted harder with its feet planted and never stepped. While the pelvis moves faster than capture_v after a shove
# the stand terms pay nothing and each touchdown is paid for the foot's travel along the motion (mjx_env capture_v).
cd "$(dirname "$0")/.."
source scripts/robot13_common.sh
CAP='capture_v=0.12 push_dv=[0.15,0.45] push_lat=0.3'
br r13f_capture   30e6 "$STAND $CAP" capture=5.0 ref_contact=1.0 scuff=-20.0 --entropy 0.01
INIT=runs/r13a_stand/params_0024M.pkl
br r13f2_capture_a 30e6 "$STAND $CAP" capture=5.0 ref_contact=1.0 scuff=-20.0 --entropy 0.01   # from the planted stand policy
echo "$(date) queue done"
