#!/bin/bash
# round 23 stage 1: k5 on each double-support table, 25M each (comparable with round 22's y2_k2: c_K2gr from k5, 25M).
# Most important first: half the double support, the ambitious one, the control, then the rest.
cd "$(dirname "$0")/.."; source scripts/robot23_common.sh
mkdir -p results/robot23
ev23 z0_y2c 1.2
L() { name=$1; T=$2; shift 2; b23 $name "$@" && ev23 $name $T; }
L d1_half   0.825 25e6 "$LANDV $D1" land_v=-100.0 $LATE
L d3_amb    0.6   25e6 "$LANDV $D3" land_v=-100.0 $LATE
L d0_ctrl   1.2   25e6 "$LANDV $D0" land_v=-100.0 $LATE
L d2_halfd  0.825 25e6 "$LANDV $D2" land_v=-100.0 $LATE
L d6_ambq   0.6   25e6 "$LANDV $D6" land_v=-100.0 $LATE
L d4_t10    1.0   25e6 "$LANDV $D4" land_v=-100.0 $LATE
L d5_zero   0.55  25e6 "$LANDV $D5" land_v=-100.0 $LATE
echo "$(date) queue done"
