#!/bin/bash
# round 19 second queue (after robot19_queue.sh): fewer gate stops on p1.
cd "$(dirname "$0")/.."; source scripts/robot19_common.sh
until grep -q "queue done" runs/robot19_queue.out; do sleep 30; done
L() { b18 "$@" && ev19 $1; }
INIT=runs/p1_s13/params_0020M.pkl
L p4_gate   20e6 "$LANDV $P1 $GATE58"  land_v=-100.0
echo "$(date) queue2 done"
