#!/bin/bash
# round 20 last runs: twins at matched steps. q6: today's leg on f1_bal's recipe for 30M (q1 had 20M). f5 / q7: each leg
# trained at a fixed 0.05-0.06 m/s command (from f1_bal / q1_v6): can either walk that fast and stay up?
cd "$(dirname "$0")/.."; source scripts/robot20_common.sh
until grep -q "queue3 done" runs/robot20_queue3.out; do sleep 30; done
L() { b20 "$@" && ev20f $1; }
F=hw_ank3_b150_m2_10dof; INIT=runs/p4_gate/params_0020M.pkl
L q6_v6   30e6 "$LANDV $Q1 $GATE58"                       land_v=-100.0 progress=3.0
F=$FB; INIT=runs/f1_bal/params_0030M.pkl
L f5_v5   30e6 "$LANDV $F1 $GATE58 cmd_vx=[0.05,0.06]"   land_v=-100.0 progress=3.0
F=hw_ank3_b150_m2_10dof; INIT=runs/q1_v6/params_0020M.pkl
L q7_v5   30e6 "$LANDV $Q1 $GATE58 cmd_vx=[0.05,0.06]"   land_v=-100.0 progress=3.0
echo "$(date) queue4 done"
