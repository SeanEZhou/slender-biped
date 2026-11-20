#!/bin/bash
# round 27 stage 1 (01:55-): from z0
cd "$(dirname "$0")/.."; source scripts/robot27_common.sh
ev27 z0_y2c
L z7_ip   25e6 "$LANDV $IP $CMDV"              land_v=-100.0 $LATE
L z7_anc  25e6 "$LANDV $IP $CMDV $ANC $STAG"   land_v=-100.0 $LATE
L z7_ancr 25e6 "$LANDV $IP $CMDV $ANCR $STAG"  land_v=-100.0 $LATE
echo "$(date) queue done"
