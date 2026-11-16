#!/bin/bash
# round 17, fifth batch (after robot17_queue4.sh), from l7_s15: the body still rolls toward the landing foot at ~23 deg/s
# and boards land at ~180 mm/s (the floor arrives before the slow set-down). l11: hover at 35 % of the lift and set down
# over the last 36 % of the swing; l12: the weight moves to the new foot over 0.6 s after touchdown (was 0.35).
cd "$(dirname "$0")/.."; source scripts/robot17_common.sh
until grep -q "queue4 done" runs/robot17_queue4.out; do sleep 30; done
L() { b17 "$@" && ev17 $1; }
INIT=runs/l7_s15/params_0020M.pkl
L l11_hover 20e6 "$LANDV $LONGU ref_table=\"c_S10hov\""     land_v=-100.0
L l12_rel   20e6 "$LANDV $LONGU ref_table=\"c_S11rel\""     land_v=-100.0
echo "$(date) queue5 done"
