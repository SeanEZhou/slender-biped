#!/bin/bash
# round 21 last run: the glide gait between p4's shift (0.13) and 0.10, with the late-shift penalty, from p4.
cd "$(dirname "$0")/.."; source scripts/robot21_common.sh
until grep -q "queue3 done" runs/robot21_queue3.out; do sleep 30; done
G115='ref_table="c_G115" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.015,0.04]'
L() { b21 "$@" && ev21 $1; }
L x4_g115L 30e6 "$LANDV $G115 $GATE58" land_v=-100.0 late_shift=-2000.0
echo "$(date) queue4 done"
