#!/bin/bash
# round 24 stage 12 (the last before the budget ends ~08:38): 0.9 s steps with a medium shift
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^/bin/bash scripts/robot24_queue11.sh" > /dev/null; do sleep 30; done
INIT=runs/h4s2_t10/params_0020M.pkl
Lg h9s_t09  0.9   20e6 "$LANDV $D9S" land_v=-100.0 $LATE
echo "$(date) queue12 done"
