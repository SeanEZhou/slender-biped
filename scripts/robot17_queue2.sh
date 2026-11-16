#!/bin/bash
# round 17, second batch (after robot17_queue.sh): the body tips toward the swing foot in single support (CPU sim:
# 26 deg/s at contact on m2 / l2 / l3) because the shift leaves the centre of mass 1 cm from the stance board's inner
# edge. shift 0.18 rad (was 0.12) puts it over the board's centre; held until touchdown.
cd "$(dirname "$0")/.."; source scripts/robot17_common.sh
until grep -q "queue done" runs/robot17_queue.out; do sleep 30; done
L() { b17 "$@" && ev17 $1; }
L l5_wide   20e6 "$LANDV $LONGU ref_table=\"c_S6wide\""     land_v=-100.0     # long soft swing + wide held shift
L l6_wshort 20e6 "$LANDV ref_table=\"c_S7wshort\""          land_v=-100.0     # k5's swing timing + wide held shift
echo "$(date) queue2 done"
