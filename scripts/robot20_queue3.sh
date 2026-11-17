#!/bin/bash
# round 20 part B follow-up: f1_bal (flat hip) holds 83-92 % up at 0.045 m/s where today's leg's q runs hold 48-58 %
# on slow hips; f3_raw (the com 10 mm behind) never walked. Push the flat hip further with q2's lesson (pay longer
# strides), and train today's leg on the same recipe and steps as the control.
cd "$(dirname "$0")/.."; source scripts/robot20_common.sh
until grep -q "queue2 done" runs/robot20_queue2.out; do sleep 30; done
evz f1_bal; evz q1_v6
L() { b20 "$@" && ev20f $1; }
F=$FB; INIT=runs/f2_fast/params_0030M.pkl
L g1_str  30e6 "$LANDV $F1F $GATE58 stride_target=0.10"   land_v=-100.0 progress=3.0 stride=100.0
F=hw_ank3_b150_m2_10dof; INIT=runs/p4_gate/params_0020M.pkl
L q4_fast 30e6 "$LANDV $Q3 $GATE58"                        land_v=-100.0 progress=3.0
F=hw_ank3_b150_m2_10dof; INIT=runs/q4_fast/params_0030M.pkl
L q5_str  30e6 "$LANDV $Q3 $GATE58 stride_target=0.10"    land_v=-100.0 progress=3.0 stride=100.0
echo "$(date) queue3 done"
