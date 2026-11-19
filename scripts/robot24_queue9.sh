#!/bin/bash
# round 24 stage 9: train gated, deploy gate-free
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name n0_lowstd" > /dev/null; do sleep 30; done
evf g0g_gate 1.2; evf g7g_t11 1.1
INIT=runs/g7g_t11/params_0020M.pkl
Lg h4g_t10  1.0   20e6 "$LANDV $D4G" land_v=-100.0 $LATE
INIT=runs/z0_y2c/params_0020M.pkl
Lg u8sg_z0  0.8   20e6 "$LANDV $U8S $GMID" land_v=-100.0 $LATE
INIT=runs/h4g_t10/params_0020M.pkl
Lg h1g_half 0.825 20e6 "$LANDV $D1G" land_v=-100.0 $LATE
echo "$(date) queue9 done"
