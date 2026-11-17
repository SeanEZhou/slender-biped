#!/bin/bash
# round 18 fifth queue (after robot18_queue4.sh): faster cadence on n7's fixed table.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until grep -q "queue4 done" runs/robot18_queue4.out; do sleep 30; done
L() { b18 "$@" && ev18 $1; }
INIT=runs/n7_s12/params_0020M.pkl
L n10_s14   20e6 "$LANDV $S14"  land_v=-100.0
echo "$(date) queue5 done"
