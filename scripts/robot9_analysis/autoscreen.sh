#!/bin/bash
# screen each round-9 branch when it finishes (vx 0.027, and 0.035 for speed)
cd "$(dirname "$0")/../.."
for r in d1b_ignore d7_vib d5_hf d6_glcp d2b_ring05 d3_ring10 d9_blind d8_glpf d10_land d4_ring15 d11_combo; do
  until [ -f runs/$r/done.json ]; do sleep 30; done
  scripts/robot9_analysis/screen9.sh $r > /dev/null 2>&1
  echo "$(date) screened $r"
done
