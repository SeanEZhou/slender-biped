#!/bin/bash
# round 24 stage 8: lower exploration noise; g0_ctrl's recipe (round 23) and d4t's, for direct comparisons
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
INIT=runs/z0_lowstd/params_0020M.pkl
Lq n0_lowstd 1.2   20e6 "$LANDV $D0 $NG" land_v=-100.0 $LATE $TERM
Lq n4_lowstd 1.0   20e6 "$LANDV $D4 $NG" land_v=-100.0 $LATE $TL
echo "$(date) queue8 done"
