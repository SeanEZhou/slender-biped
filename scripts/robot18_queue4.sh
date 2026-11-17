#!/bin/bash
# round 18 fourth queue (after robot18_queue3.sh): which part of the walk tables breaks the balance?
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until grep -q "queue3 done" runs/robot18_queue3.out; do sleep 30; done
L() { b18 "$@" && ev18 $1; }
L n8_s13ik   20e6 "$LANDV $S13"  land_v=-100.0
L n9_w7bump  20e6 "$LANDV $W7"   land_v=-100.0
echo "$(date) queue4 done"
