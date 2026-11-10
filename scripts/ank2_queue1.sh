#!/bin/bash
# 2026-11-09 ankle-roll leg as designed (hw_ank2_*): pad comparison on the flat-foot shift / hike / step reference.
cd "$(dirname "$0")/.."
for pad in r80 b150 foot r60 d70 b100; do
  EXTRA_SET='dr_floor_tc=[0.02,0.8]' scripts/b150_run.sh a2s_${pad} ank2_step hw_ank2_${pad}_legs_10dof 20e6
done
EXTRA_SET='dr_floor_tc=[0.02,0.8]' scripts/b150_run.sh a2b_r80 ank2_bal hw_ank2_r80_legs_10dof 20e6
