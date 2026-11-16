#!/bin/bash
# Round 14 lane 1 (restarted 13:28 with --reset-cmd: k5's sideways-command input had std 0 and untrained weights, so
# the step command arrived as a huge random input). Stage A on rs1 (1.8 s steps), then stage B from it if time allows.
cd "$(dirname "$0")/.."
source scripts/robot14_common.sh
br r14c_steps 24e6 "$STEPA" --reset-cmd 0.03,0.02,1.0
echo "$(date) lane done"
