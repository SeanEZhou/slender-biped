#!/bin/bash
# round 21: the continuous (glide) gait from p4, at three shift amounts, and with small sideways shoves.
cd "$(dirname "$0")/.."; source scripts/robot21_common.sh
L() { b21 "$@" && ev21 $1; }
L w1_g13   30e6 "$LANDV $G13 $GATE58"          land_v=-100.0
L w2_g10   30e6 "$LANDV $G10 $GATE58"          land_v=-100.0
L w3_g08   30e6 "$LANDV $G08 $GATE58"          land_v=-100.0
L w4_g10s  30e6 "$LANDV $G10 $GATE58 $SHOVE"   land_v=-100.0
echo "$(date) queue done"
