#!/bin/bash
# round 29 queue 2 (01:50): queue 1 stopped after z9_a1m's training so the turning reference gets checked in between
cd "$(dirname "$0")/.."; source scripts/robot29_common.sh
until [ -f runs/z9_a1m/done.json ]; do sleep 20; done; echo "$(date) end z9_a1m"
ev29 z9_a1m
TE="scripts/gpu_python.sh scripts/robot29/turn_eval.py"
{ $TE z9_a1m --zero --n 16 --set turn_yaw=true; $TE z9_a1m --zero --n 16 --deg 90 --set turn_yaw=true
  $TE z9_a1m --n 16; $TE z9_a1m --n 16 --set turn_yaw=true; $TE z0_y2c --n 16 --set walk_state=true 'ref_table="c_K2ip"' turn_yaw=true; } 2>&1 | grep -vi "$FLT" > results/robot29/turncheck.txt
echo "$(date) turn check done"
L29 z9_a5m 20e6 "$START" land_v=-100.0 $LATE --anchor 5.0 --mirror 1.0
echo "$(date) queue2 done"
