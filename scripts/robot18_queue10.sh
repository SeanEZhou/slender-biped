#!/bin/bash
# round 18 tenth queue (after robot18_queue9.sh): n15's table at 1.1 s steps, from n15. Then the final ranking.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until grep -q "queue9 done" runs/robot18_queue9.out; do sleep 30; done
L() { b18 "$@" && ev18 $1; }
INIT=runs/n15_s20/params_0020M.pkl
L n19_s25   20e6 "$LANDV $S25"  land_v=-100.0
echo "$(date) queue10 done"
