#!/bin/bash
# round 24 stage 13 (the last slot, ends ~08:30): h4x +20M (h4s's second 20M took it from 48 to 75-81 % gate-free)
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^/bin/bash scripts/robot24_queue12.sh" > /dev/null; do sleep 30; done
INIT=runs/h4x_t10/params_0020M.pkl
Lg h4x2_t10 1.0   20e6 "$LANDV $D4S6" land_v=-100.0 $LATE
echo "$(date) queue13 done"
