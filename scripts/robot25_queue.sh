#!/bin/bash
# round 25 stage 1: one change at a time from z0, then the three together.
cd "$(dirname "$0")/.."; source scripts/robot25_common.sh
# ev25 z0_y2c (done 12:40)
L z5_hold  20e6 "$LANDV $K2 $HOLD"               land_v=-100.0 $LATE
L z5_mir   20e6 "$LANDV $K2 $MIR"                land_v=-100.0 $LATE
L z5_shv   20e6 "$LANDV $K2 $SHOVE"              land_v=-100.0 $LATE
L z5_hms   20e6 "$LANDV $K2 $HOLD $MIR $SHOVE"   land_v=-100.0 $LATE
echo "$(date) queue done"
