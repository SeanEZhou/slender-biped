#!/bin/bash
# round 18 second batch (replaces robot18_queue.sh after n2_w10): the double-support sway first, then n3 / n4.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until [ -f runs/n2_w10/done.json ] && ! pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name n2_w10" >/dev/null; do sleep 30; done
echo "$(date) end n2_w10"; ev18 n2_w10
L() { b18 "$@" && ev18 $1; }
L n5_w12d  20e6 "$LANDV $W5"   land_v=-100.0
L n6_w10d  20e6 "$LANDV $W6"   land_v=-100.0
echo "$(date) queue2 done"
