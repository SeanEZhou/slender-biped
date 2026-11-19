#!/bin/bash
# round 24 stage 5: the tilt penalty 10x lighter, z0 onto the ramped stride tables, gate off
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
Lq u8r_t    0.8   20e6 "$LANDV $R8 $NG" land_v=-100.0 $LATE $TL
Lq s4hr_t   0.5   20e6 "$LANDV $R4H $NG" land_v=-100.0 $LATE $TL
echo "$(date) queue5 done"
