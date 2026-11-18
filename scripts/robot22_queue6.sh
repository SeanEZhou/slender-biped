#!/bin/bash
# round 22: y7 (c_Z1soft from k5: 88 % up, loose 33 %, softer landings) continued 20M on its own table.
cd "$(dirname "$0")/.."; source scripts/robot22_common.sh
while pgrep -f "^/bin/bash scripts/robot22_queue5.sh" > /dev/null; do sleep 20; done
L() { b22 "$@" && ev22 $1; }
INIT=runs/y7_z1/params_0025M.pkl
L z7_y7c 20e6 "$LANDV $Z1"   land_v=-100.0 $LATE
echo "$(date) queue6 done"
