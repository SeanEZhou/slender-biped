#!/bin/bash
# round 24 stage 3: the ramped stride tables (gait start), gate-free and gated branches
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name u8_z0" > /dev/null; do sleep 30; done
dg u8_z0 u8_z0 --vx 0.0375
Lq u8r_z0   0.8   20e6 "$LANDV $R8 $NG" land_v=-100.0 $LATE
Lq u8rg_z0  0.8   20e6 "$LANDV $R8 $G8" land_v=-100.0 $LATE
echo "$(date) queue3 done"
