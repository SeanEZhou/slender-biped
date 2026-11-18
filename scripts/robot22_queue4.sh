#!/bin/bash
# round 22 stage 3: from k5 at 25M (as stage 1): y2's table with a second seed, a later release, the slower descent.
cd "$(dirname "$0")/.."; source scripts/robot22_common.sh
while pgrep -f "^/bin/bash scripts/robot22_queue[23].sh" > /dev/null; do sleep 20; done
L() { b22 "$@" && ev22 $1; }
L y2b_k2s1 25e6 "$LANDV $K2"   land_v=-100.0 $LATE --seed 1
L y6_k6    25e6 "$LANDV $K6"   land_v=-100.0 $LATE
L y7_z1    25e6 "$LANDV $Z1"   land_v=-100.0 $LATE
echo "$(date) queue4 done"
