#!/bin/bash
# round 24 stage 2: the cadence ladder, gated and gate-free branches (see robot24_common.sh)
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name s4h_z0" > /dev/null; do sleep 30; done
ev24 s4h_z0 0.5; dg s4h_z0 s4h_z0 --vx 0.06
L u8_z0     0.8   20e6 "$LANDV $U8 $NG" land_v=-100.0 $LATE
L u8g_z0    0.8   20e6 "$LANDV $U8 $G8" land_v=-100.0 $LATE
INIT=runs/u8_z0/params_0020M.pkl
L u65_u8    0.65  20e6 "$LANDV $U65 $NG" land_v=-100.0 $LATE
INIT=runs/u8g_z0/params_0020M.pkl
L u65g_u8g  0.65  20e6 "$LANDV $U65 $G65" land_v=-100.0 $LATE
INIT=runs/u65_u8/params_0020M.pkl
L s4h_u65   0.5   20e6 "$LANDV $S4H $NG" land_v=-100.0 $LATE
echo "$(date) queue2 done"
