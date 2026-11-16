#!/bin/bash
# round 18 first batch, one lane at a time (WSL 7.9 GB): the four continuous tables, 20M each from l9_s18 20M.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
L() { b18 "$@" && ev18 $1; }
L n1_w12   20e6 "$LANDV $W1"   land_v=-100.0
L n2_w10   20e6 "$LANDV $W2"   land_v=-100.0
L n3_w12l  20e6 "$LANDV $W3"   land_v=-100.0
L n4_w10f  20e6 "$LANDV $W4"   land_v=-100.0
echo "$(date) queue done"
