#!/bin/bash
# round 22 stage 2: y2's table varied (landing descent, shift lead, cadence, release timing), from y2 at 25M, 20M each.
cd "$(dirname "$0")/.."; source scripts/robot22_common.sh
while pgrep -f "^/bin/bash scripts/robot22_queue.sh" > /dev/null; do sleep 20; done
INIT=runs/y2_k2/params_0025M.pkl
L() { b22 "$@" && ev22 $1; }
L z1_soft  20e6 "$LANDV $Z1"   land_v=-100.0 $LATE
L z4_r08   20e6 "$LANDV $Z4"   land_v=-100.0 $LATE
L z3_t11   20e6 "$LANDV $Z3"   land_v=-100.0 $LATE
L z2_lead  20e6 "$LANDV $Z2"   land_v=-100.0 $LATE
echo "$(date) queue2 done"
