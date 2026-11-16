#!/bin/bash
# round 17, sixth batch (after robot17_queue5.sh): the landing penalty mattered on the soft-swing table (l8 without it:
# harder landings, loose sim 0-5 %); twice as much on l7.
cd "$(dirname "$0")/.."; source scripts/robot17_common.sh
until grep -q "queue5 done" runs/robot17_queue5.out; do sleep 30; done
L() { b17 "$@" && ev17 $1; }
INIT=runs/l7_s15/params_0020M.pkl
L l13_lv2   20e6 "$LANDV $LONGU ref_table=\"c_S8mid15\""    land_v=-200.0
echo "$(date) queue6 done"
