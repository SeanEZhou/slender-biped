#!/bin/bash
# round 22: the second seed on y2's table, continued 20M on its own table (as z0 was for y2).
cd "$(dirname "$0")/.."; source scripts/robot22_common.sh
while pgrep -f "^/bin/bash scripts/robot22_queue4.sh" > /dev/null; do sleep 20; done
L() { b22 "$@" && ev22 $1; }
INIT=runs/y2b_k2s1/params_0025M.pkl
L z0b_y2bc 20e6 "$LANDV $K2"   land_v=-100.0 $LATE --seed 1
echo "$(date) queue5 done"
