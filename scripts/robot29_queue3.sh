#!/bin/bash
# round 29 queue 3 (02:20). z9_a1m (anchor 1, mirror 1) kept z0's walking (92-94 % up, 34-43 cm), its start (+13 deg
# shift at the first lift-off) and stops (29-30 of 32), symmetric (shift +9.5 / +9.5); weak with loose hips (12 % gated).
# The turning reference turns the right way (57-59 deg of 60 with z9_a1m untrained, but half fall).
cd "$(dirname "$0")/.."; source scripts/robot29_common.sh
until grep -q "queue2 done" runs/robot29_queue.log; do sleep 20; done
late() { [ $(date +%H%M) -gt 0845 ] && [ $(date +%H%M) -lt 1200 ] && { echo "$(date) past 08:45, not starting $1"; return 0; }; return 1; }
INIT=runs/z9_a1m/params_0020M.pkl
late z9_t1 || L29t z9_t1 25e6 "$START $TURNY" land_v=-100.0 $LATE --anchor 1.0 --mirror 1.0      # turning, with the reference
INIT=runs/z0_y2c/params_0020M.pkl
late z9_ng || L29 z9_ng 20e6 "$START $NG" land_v=-100.0 $LATE --anchor 1.0 --mirror 1.0          # start/stop, trained gate-free
INIT=runs/z9_a1m/params_0020M.pkl
late z9_t0 || L29t z9_t0 25e6 "$START $TURN" land_v=-100.0 $LATE --anchor 1.0 --mirror 1.0       # turning, the policy's own
echo "$(date) queue3 done"
