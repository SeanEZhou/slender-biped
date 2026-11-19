#!/bin/bash
# round 24 stage 10: a smaller shift at faster cadences, train gated, deploy gate-free
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name u8sg_z0" > /dev/null; do sleep 30; done
evf u8sg_z0 0.8
INIT=runs/g7g_t11/params_0020M.pkl
Lg h4s_t10  1.0   20e6 "$LANDV $D4S" land_v=-100.0 $LATE
INIT=runs/h4s_t10/params_0020M.pkl
Lg h1s_half 0.825 20e6 "$LANDV $D1S" land_v=-100.0 $LATE
echo "$(date) queue10 done"
