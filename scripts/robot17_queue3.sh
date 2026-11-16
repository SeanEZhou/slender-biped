#!/bin/bash
# round 17, third batch (after robot17_queue2.sh). l4_mid (c_S5mid: 0.6 s soft swing, held shift, lift 0.60) is the
# best so far; l5_wide (shift 0.18) had the softest landings but fell sideways in 20M from m2.
cd "$(dirname "$0")/.."; source scripts/robot17_common.sh
until grep -q "queue2 done" runs/robot17_queue2.out; do sleep 30; done
L() { b17 "$@" && ev17 $1; }
INIT=runs/l4_mid/params_0020M.pkl
L l7_s15    20e6 "$LANDV $LONGU ref_table=\"c_S8mid15\""    land_v=-100.0     # l4 + an in-between shift 0.15
L l4b_mid   20e6 "$LANDV $LONGU ref_table=\"c_S5mid\""      land_v=-100.0     # l4 + 20M
INIT=runs/m2_attland/params_0014M.pkl
L l8_nolv   20e6 "$LONGU ref_table=\"c_S5mid\""                               # l4 without the landing penalty
echo "$(date) queue3 done"
