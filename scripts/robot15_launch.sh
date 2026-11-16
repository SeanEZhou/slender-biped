#!/bin/bash
# Round 15: at most TWO lanes, started one after another (each once the previous has compiled). WSL has 7.9 GB RAM:
# three compiles at once hung it (19:39), and two trainings plus a third compile ran it out of memory (19:48 kernel
# page allocation failures; hung until 20:41). Lane 3 (r15b_push) is left for a later session.
cd "$(dirname "$0")/.."
wait_compiled() { for i in $(seq 1 90); do grep -q "step     0.0M" runs/$1.log 2>/dev/null && return; sleep 10; done; }
scripts/robot15_lane2.sh > runs/robot15_lane2.out 2>&1 &
sleep 20; wait_compiled r15w_walk
scripts/robot15_lane1.sh > runs/robot15_lane1.out 2>&1 &
wait
