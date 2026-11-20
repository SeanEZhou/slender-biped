#!/bin/bash
# round 26 pilot (22:30-23:30): fine-tunes from z0 (z0's recipe), 15M each
cd "$(dirname "$0")/.."; source scripts/robot26_common.sh
L z6_er 15e6 "$LANDV $K2 $ERR $RELAX"   land_v=-100.0 $LATE      # errors + relaxed reference
L z6_e  15e6 "$LANDV $K2 $ERR"          land_v=-100.0 $LATE      # errors only
[ $(date +%H%M) -gt 2310 ] && { echo "$(date) past 23:10, not starting z6_er2"; exit; }
L z6_er2 10e6 "$LANDV $K2 $ERR2 $RELAX" land_v=-100.0 $LATE      # larger errors + relaxed reference
echo "$(date) queue done"
