#!/bin/bash
# round 22 stage 2 control: y2 continued 20M on its own table (c_K2gr), so the z runs compare like for like.
cd "$(dirname "$0")/.."; source scripts/robot22_common.sh
while pgrep -f "^/bin/bash scripts/robot22_queue2.sh" > /dev/null; do sleep 20; done
INIT=runs/y2_k2/params_0025M.pkl
L() { b22 "$@" && ev22 $1; }
L z0_y2c   20e6 "$LANDV $K2"   land_v=-100.0 $LATE
echo "$(date) queue3 done"
