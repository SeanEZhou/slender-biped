#!/bin/bash
# 22:40: two k5 lanes grow to ~3.9 GB each (7.9 GB RAM): m2 ran out of room (mem_guard killed it). It runs after m1,
# alone, shortened to 14M to fit the 90 min.
cd "$(dirname "$0")/.."
source scripts/robot16_common.sh
while [ ! -f runs/robot16_eval1.done ]; do sleep 20; done   # after m1 and its sim tests (alone)
b16 m2_attland 14e6 "$ATT" foot_tilt=-30.0
