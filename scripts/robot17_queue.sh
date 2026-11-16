#!/bin/bash
# round 17 lanes, one at a time (a k5-type train.py grows to ~3.9 GB; WSL has 7.9 GB). Each 20M from m2 14M, then evals.
cd "$(dirname "$0")/.."; source scripts/robot17_common.sh
L() { b17 "$@" && ev17 $1; }
L l1_landv  20e6 "$LANDV"                                  land_v=-100.0
L l2_long   20e6 "$LANDV $LONGU ref_table=\"c_S3long\""     land_v=-100.0
L l3_lhold  20e6 "$LANDV $LONGU ref_table=\"c_S4lh\""       land_v=-100.0
L l4_mid    20e6 "$LANDV $LONGU ref_table=\"c_S5mid\""      land_v=-100.0
echo "$(date) queue done"
