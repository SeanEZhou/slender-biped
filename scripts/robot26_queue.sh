#!/bin/bash
# round 26 pilot (22:30-23:30): fine-tunes from z0 (z0's recipe), 15M each
cd "$(dirname "$0")/.."; source scripts/robot26_common.sh
L z6_er 15e6 "$LANDV $K2 $ERR $RELAX"   land_v=-100.0 $LATE      # errors + relaxed reference
L z6_e  15e6 "$LANDV $K2 $ERR"          land_v=-100.0 $LATE      # errors only
[ $(date +%H%M) -le 2310 ] && L z6_es 10e6 "$LANDV $K2 step_err=[0.5,0.02,0.03] $RELAX" land_v=-100.0 $LATE  # smaller errors + relaxed
ev26 z6_er   # re-evaluate (its first evals had the errors on throughout)
echo "$(date) queue done"
