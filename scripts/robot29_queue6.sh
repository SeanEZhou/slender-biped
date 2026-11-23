#!/bin/bash
# round 29 queue 6 (07:10): z9_as5 (z7_anc2's reward-only anchoring + staggered starts, anchor 5) is the steadiest
# start/stop policy (96 % gated at every speed, 31 of 32 through start/stop): turning from it, as z9_t2 from z9_a5m
cd "$(dirname "$0")/.."; source scripts/robot29_common.sh
until grep -q "queue5 done" runs/robot29_queue.log; do sleep 20; done
late() { [ $(date +%H%M) -gt 0845 ] && [ $(date +%H%M) -lt 1200 ] && { echo "$(date) past 08:45, not starting $1"; return 0; }; return 1; }
INIT=runs/z9_as5/params_0020M.pkl
late z9_t5 || L29t z9_t5 25e6 "$START $ANCW $STAG $TURNY" land_v=-100.0 $LATE --anchor 5.0 --mirror 1.0
echo "$(date) queue6 done"
