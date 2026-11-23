#!/bin/bash
# round 29 queue 5 (06:10). Turning costs some walking: z9_t1 81-94 % up, z9_t2 (from z9_a5m, anchor 5) 88-96 %,
# against z9_a5m 92-96 %. Two milder turning recipes from z9_a5m: fewer turns (15 % of the nonzero commands), or a
# gentler turning reference (scale 0.5: up to 0.15 rad/s, hip yaw +-5 deg).
cd "$(dirname "$0")/.."; source scripts/robot29_common.sh
until grep -q "queue4 done" runs/robot29_queue.log; do sleep 20; done
late() { [ $(date +%H%M) -gt 0845 ] && [ $(date +%H%M) -lt 1200 ] && { echo "$(date) past 08:45, not starting $1"; return 0; }; return 1; }
INIT=runs/z9_a5m/params_0020M.pkl
late z9_t3 || L29t z9_t3 30e6 "$START turn_prob=0.15 turn_wz=[0.25,0.8] turn_yaw=true turn_yaw_scale=0.67" land_v=-100.0 $LATE --anchor 5.0 --mirror 1.0
late z9_t4 || L29t z9_t4 30e6 "$START $TURN turn_yaw=true turn_yaw_scale=0.5" land_v=-100.0 $LATE --anchor 5.0 --mirror 1.0
echo "$(date) queue5 done"
