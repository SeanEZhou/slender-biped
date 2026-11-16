#!/bin/bash
# round 17, fourth batch (after robot17_queue3.sh): l7_s15 (shift 0.15, from l4_mid) is the most robust yet (loose sim
# 40 % / 52 %); widen the shift one more step from it (a curriculum: l5's 0.18 straight from m2 fell sideways), and l7 with the body-jolt
# penalty doubled (l4b_mid showed 20M more on the same table adds nothing).
cd "$(dirname "$0")/.."; source scripts/robot17_common.sh
until grep -q "queue3 done" runs/robot17_queue3.out; do sleep 30; done
L() { b17 "$@" && ev17 $1; }
INIT=runs/l7_s15/params_0020M.pkl
L l9_s18    20e6 "$LANDV $LONGU ref_table=\"c_S9mid18\""    land_v=-100.0     # l7 -> shift 0.18
L l10_jolt  20e6 "$LANDV $LONGU ref_table=\"c_S8mid15\""    land_v=-100.0 ang_acc=-0.012   # l7 + body-jolt penalty x2
echo "$(date) queue4 done"
