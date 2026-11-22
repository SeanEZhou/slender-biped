#!/bin/bash
# Round 28: reproduce every fall of Nov 19-20 with the policy that ran (results/robot28/repro_<run>_logged.json)
cd "$(dirname "$0")/../.."
for r in z7_anc2 z0_y2c z6_e z7_rgq z7_var z5_shv10 z5_shv2 k5_high2 h4s2_t10 h4x2_t10 h9s_t09 g0g_gate g7g_t11; do
  scripts/gpu_python.sh scripts/robot28/repro.py $r --controls 2 2>&1 | grep -vi 'warn\|warp\|^W[0-9]'
done > results/robot28/repro_all.log
echo "$(date) repro_all done" >> results/robot28/repro_all.log
