#!/bin/bash
# round 21 second queue. w3_g08 (glide, shift 0.08) is the best of queue 1 in simulation (96-98 % up, 41-48 cm, 96 %
# through shoves) but lifts with 67 % of its peak leg spread: calibrate the shift-timing check on p4 (works on the
# robot) and q7 (shifted late, fell), then: w3 + shoves, a second seed of w3, and the glide gait on leg v2.
cd "$(dirname "$0")/.."; source scripts/robot21_common.sh
until grep -q "queue done" runs/robot21_queue.out; do sleep 30; done
for r in p4_gate q7_v5 q2_v6s w1_g13 w2_g10; do evq $r; done
ev21 p4_gate
L() { b21 "$@" && ev21 $1; }
INIT=runs/w3_g08/params_0030M.pkl
L w5_g08s  20e6 "$LANDV $G08 $GATE58 $SHOVE"   land_v=-100.0
INIT=runs/p4_gate/params_0020M.pkl
L w6_g08b  30e6 "$LANDV $G08 $GATE58"          land_v=-100.0 --seed 1
F=$FB; INIT=runs/f1_bal/params_0030M.pkl
L f6_g07   30e6 "$LANDV $FG07 $GATE58"         land_v=-100.0
echo "$(date) queue2 done"
