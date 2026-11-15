#!/bin/bash
# screen each round-10 branch when it finishes (vx 0.02, run-time filter 0.6)
cd "$(dirname "$0")/../.."
for r in k1_speed k2_glcp k4_high o1_hold4 k6_combo o3_hold4g k3_com o2_hold2 k5_high2; do
  until [ -f runs/$r/done.json ] || [ -f runs/robot10_stop ]; do sleep 30; done
  [ -f runs/$r/done.json ] || break
  scripts/robot10_analysis/screen10.sh $r > /dev/null 2>&1
  echo "$(date) screened $r"
done
