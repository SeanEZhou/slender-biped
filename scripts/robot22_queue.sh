#!/bin/bash
# round 22 stage 1: the benchmarks (k5, l9) on today's evaluation, then k5 on each table, 25M each.
cd "$(dirname "$0")/.."; source scripts/robot22_common.sh
ev22 k5_high2; ev22 l9_s18
L() { b22 "$@" && ev22 $1; }
L y3_k3   25e6 "$LANDV $K3"   land_v=-100.0 $LATE
L y2_k2   25e6 "$LANDV $K2"   land_v=-100.0 $LATE
L y1_k1   25e6 "$LANDV $K1"   land_v=-100.0 $LATE
L y5_k5   25e6 "$LANDV $K5"   land_v=-100.0 $LATE
L y4_k4   25e6 "$LANDV $K4"   land_v=-100.0 $LATE
echo "$(date) queue done"
