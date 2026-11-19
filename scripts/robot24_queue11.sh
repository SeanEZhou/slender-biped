#!/bin/bash
# round 24 stage 11: continue the smaller-shift 1.0 s policy; a smaller shift still
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^/bin/bash scripts/robot24_queue10.sh" > /dev/null; do sleep 30; done
INIT=runs/h4s_t10/params_0020M.pkl
Lg h4s2_t10 1.0   20e6 "$LANDV $D4S" land_v=-100.0 $LATE
INIT=runs/g7g_t11/params_0020M.pkl
Lg h4x_t10  1.0   20e6 "$LANDV $D4S6" land_v=-100.0 $LATE
echo "$(date) queue11 done"
