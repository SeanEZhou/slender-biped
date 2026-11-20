#!/bin/bash
# round 25 stage 4 (written 16:05): the control (z5_ctl) shows 20M more of z0's recipe loses robustness by itself;
# against it the shoves help (z5_shv keeps z0's level). Build on the shoves: harder states with them (a sideways mass
# offset, larger start tilt), and the shoves continued from z5_shv at the larger size.
cd "$(dirname "$0")/.."; source scripts/robot25_common.sh
while pgrep -f "^/bin/bash scripts/robot25_queue3.sh" > /dev/null; do sleep 20; done
INIT=runs/z0_y2c/params_0020M.pkl
L z5_shh   20e6 "$LANDV $K2 $SHOVE $HARD"    land_v=-100.0 $LATE
INIT=runs/z5_shv/params_0020M.pkl
L z5_shv2  20e6 "$LANDV $K2 $SHOVE10"        land_v=-100.0 $LATE
echo "$(date) queue4 done"
