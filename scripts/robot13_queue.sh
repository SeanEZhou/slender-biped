#!/bin/bash
# Round 13 queue 1 (overnight 2026-11-15): four ways to recover from shoves, all from k5_high2.
cd "$(dirname "$0")/.."
source scripts/robot13_common.sh
while ps -eo args | grep -qE "^[^ ]*python[^ ]* scripts/train.py"; do sleep 30; done
br r13a_stand      24e6 "$STAND push_dv=[0.05,0.3]"                        # stand only, shoves 0.05-0.3 m/s; k5's rewards (stand = ref pose, both feet down)
br r13c_sup        30e6 "push_dv=[0.03,0.25] $SUP"                         # walk + stand with shoves; the supervisor stops the walk on a predicted tip
br r13b_walkpush   24e6 "push_dv=[0.03,0.2]"                               # walk + stand with shoves, no supervisor (a robust walker)
br r13a2_stand_step 24e6 "$STAND push_dv=[0.1,0.4] push_lat=0.5" ref_contact=1.0 scuff=-20.0   # bigger shoves, half sideways, steps cheaper
echo "$(date) queue done"
