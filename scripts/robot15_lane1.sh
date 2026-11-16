#!/bin/bash
# Round 15 lane 1: stage A on rs3 (random decisive steps every ~1-2.5 s at rest), from r14c2_slow (stage A on rs2).
cd "$(dirname "$0")/.."
source scripts/robot15_common.sh
bs r15a_steps 8e6 runs/r14c2_slow/params_0024M.pkl "rec_rand=1.0 rec_rest_s=[0.8,2.0]"
echo "$(date) lane done"
