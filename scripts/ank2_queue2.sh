#!/bin/bash
# 2026-11-09 pad comparison, round 2: the reward overrides of the successful ankle study (b150A_*): no progress term
# (it paid sliding the feet forward), stride x60, ref_swing x15, slip -300.
cd "$(dirname "$0")/.."
while pgrep -f "train.py --name a2s_r80 " > /dev/null; do sleep 20; done
STEPRW="progress=0.0 stride=60.0 ref_swing=15.0 slip=-300.0"
for pad in r80 b150 foot r60 d70 b100; do
  EXTRA_SET='dr_floor_tc=[0.02,0.8]' scripts/b150_run.sh a2t_${pad} ank2_step hw_ank2_${pad}_legs_10dof 20e6 --rw $STEPRW
done
EXTRA_SET='dr_floor_tc=[0.02,0.8]' scripts/b150_run.sh a2b_r80 ank2_bal hw_ank2_r80_legs_10dof 20e6 --rw slip=-300.0
