#!/bin/bash
# 20:47: back to two lanes (as every round before 15): the walking lane again once r15a_steps has compiled.
cd "$(dirname "$0")/.."
for i in $(seq 1 90); do grep -q "step     0.0M" runs/r15a_steps.log 2>/dev/null && break; sleep 10; done
scripts/robot15_lane2.sh > runs/robot15_lane2.out 2>&1
