#!/bin/bash
# round 24 stage 6: round 23's continuous-sweep tables, gate off, the light tilt penalty
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
Lq d4t_t10  1.0   20e6 "$LANDV $D4 $NG" land_v=-100.0 $LATE $TL
INIT=runs/d4t_t10/params_0020M.pkl
Lq d1t_half 0.825 20e6 "$LANDV $D1 $NG" land_v=-100.0 $LATE $TL
echo "$(date) queue6 done"
