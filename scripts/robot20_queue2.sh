#!/bin/bash
# round 20 part B: the flat-hip leg, from p4 (same policy inputs and outputs), on p4's recipe.
cd "$(dirname "$0")/.."; source scripts/robot20_common.sh
until grep -q "queue done" runs/robot20_queue.out; do sleep 30; done
L() { b20 "$@" && ev20f $1; }
F=$FB; L f1_bal   30e6 "$LANDV $F1 $GATE58"                          land_v=-100.0 progress=3.0
F=$FB; L f2_fast  30e6 "$LANDV $F1 $GATE58 cmd_vx=[0.03,0.08]"      land_v=-100.0 progress=3.0
F=$FR; L f3_raw   30e6 "$LANDV $F1 $GATE58"                          land_v=-100.0 progress=3.0
F=$FB; L f4_t11   30e6 "$LANDV $F2 $GATE58"                          land_v=-100.0 progress=3.0
echo "$(date) queue2 done"
