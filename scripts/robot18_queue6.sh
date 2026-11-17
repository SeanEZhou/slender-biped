#!/bin/bash
# round 18 sixth queue (after robot18_queue5.sh): retraction while high, the trip fix that keeps l9's stability?
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until grep -q "queue5 done" runs/robot18_queue5.out; do sleep 30; done
L() { b18 "$@" && ev18 $1; }
INIT=runs/n7_s12/params_0020M.pkl
L n11_s16r   20e6 "$LANDV $S16"  land_v=-100.0
echo "$(date) queue6 done"
