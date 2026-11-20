#!/bin/bash
# round 25 stage 2: z5_hold (551 -> 503) and z5_mir (537 -> 454) both degraded in training. Control: z0 continued on
# its own recipe, nothing changed, to tell a change's cost from fine-tuning drift.
cd "$(dirname "$0")/.."; source scripts/robot25_common.sh
while pgrep -f "^/bin/bash scripts/robot25_queue.sh" > /dev/null; do sleep 20; done
L z5_ctl   20e6 "$LANDV $K2"   land_v=-100.0 $LATE
echo "$(date) queue2 done"
