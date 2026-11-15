#!/bin/bash
# Round 11 queue (~90 min of GPU), most informative first; the tail can be cut.
cd "$(dirname "$0")/.."
source scripts/robot11_common.sh
while ps -eo args | grep -qE "^[^ ]*python[^ ]* scripts/train.py"; do sleep 30; done
br p2_side_gate 24e6 "$SIDE $RGATE"   # the target: k5 steps, sideways open loop, live slow pitch, roll gate, loose hips
br m2_gate      16e6 "$RGATE"         # k5 + roll gate + loose hips (closed loop)
br p1_side      20e6 "$SIDE"          # k5 + sideways open loop + loose hips, no roll gate
br m1_loose     12e6 ""               # k5 + loose hips only (control)
echo "$(date) queue done"
