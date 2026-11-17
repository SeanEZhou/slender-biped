#!/bin/bash
# round 18 third queue (replaces robot18_queue2.sh after n5_w12d): n6 (W6) only if n5's double-support sway holds up
# in the robot sim (>= 60 %); l9 + the release fix (n7) either way.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until [ -f runs/n5_w12d/done.json ] && ! pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name n5_w12d" >/dev/null; do sleep 30; done
echo "$(date) end n5_w12d"; ev18 n5_w12d
L() { b18 "$@" && ev18 $1; }
u=$(up_robot n5_w12d); echo "$(date) n5 robot-sim up ${u}%"
if [ "${u:-0}" -ge 60 ]; then
  L n6_w10d  20e6 "$LANDV $W6"   land_v=-100.0
  L n7_s12   20e6 "$LANDV $S12"  land_v=-100.0
else
  L n7_s12   20e6 "$LANDV $S12"  land_v=-100.0
fi
echo "$(date) queue3 done"
