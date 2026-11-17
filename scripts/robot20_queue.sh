#!/bin/bash
# round 20 part A: farther p4 (from p4_gate at 20M, its recipe: p1 + relaxed gate + land_v).
cd "$(dirname "$0")/.."; source scripts/robot20_common.sh
L() { b20 "$@" && ev20 $1; }
L q1_v6   20e6 "$LANDV $Q1 $GATE58"                         land_v=-100.0 progress=3.0
L q2_v6s  20e6 "$LANDV $Q1 $GATE58 stride_target=0.07"      land_v=-100.0 progress=3.0 stride=100.0
L q3_t11  20e6 "$LANDV $Q2 $GATE58"                         land_v=-100.0 progress=3.0
echo "$(date) queue done"
