#!/bin/bash
# round 19 (2 h GPU from 19:20): three lanes, one at a time.
cd "$(dirname "$0")/.."; source scripts/robot19_common.sh
L() { b18 "$@" && ev19 $1; }
INIT=runs/n15_s20/params_0020M.pkl
L p1_s13    20e6 "$LANDV $P1"  land_v=-100.0
INIT=runs/n7_s12/params_0020M.pkl
L p3_bump   20e6 "$LANDV $P3"  land_v=-100.0
INIT=runs/n15_s20/params_0020M.pkl
L p2_s18    20e6 "$LANDV $P2"  land_v=-100.0
echo "$(date) queue done"
