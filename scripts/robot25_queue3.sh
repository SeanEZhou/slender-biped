#!/bin/bash
# round 25 stage 3 (written 14:45): z5_mir fixed z0's lopsidedness (stance level, both feet landing 121-132 mm apart,
# the first left step forward) but lost stability in 20M (52-62 % up): it has to relearn half its behaviour. More of it.
# And larger shoves: z5_shv (up to 0.06 m/s) didn't help against 0.2 m/s.
cd "$(dirname "$0")/.."; source scripts/robot25_common.sh
while pgrep -f "^/bin/bash scripts/robot25_queue2?.sh" > /dev/null; do sleep 20; done
INIT=runs/z5_mir/params_0020M.pkl
L z5_mir2  30e6 "$LANDV $K2 $MIR"            land_v=-100.0 $LATE
INIT=runs/z0_y2c/params_0020M.pkl
L z5_shv10 20e6 "$LANDV $K2 $SHOVE10"        land_v=-100.0 $LATE
echo "$(date) queue3 done"
