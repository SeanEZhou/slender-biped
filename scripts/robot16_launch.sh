#!/bin/bash
# Round 16: two lanes (as every round), the second once the first has compiled.
cd "$(dirname "$0")/.."
source scripts/robot16_common.sh
wait_compiled() { for i in $(seq 1 90); do grep -q "step     0.0M" runs/$1.log 2>/dev/null && return; sleep 10; done; }
b16 m1_attgate 20e6 "$ATT" > runs/robot16_lane1.out 2>&1 &
sleep 20; wait_compiled m1_attgate
b16 m2_attland 20e6 "$ATT" foot_tilt=-30.0 > runs/robot16_lane2.out 2>&1 &   # + boards landing flat (foot_tilt x3)
wait
