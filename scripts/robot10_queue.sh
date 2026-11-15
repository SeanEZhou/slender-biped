#!/bin/bash
# Round 10 queue, one GPU job at a time, most informative first (the tail can be cut if time runs out).
cd "$(dirname "$0")/.."
source scripts/robot10_common.sh
while ps -eo args | grep -qE "^[^ ]*python[^ ]* scripts/train.py"; do sleep 30; done
br k1_speed  16e6 "$SPEED"                       # make the step target follow the speed command
br k2_glcp   16e6 ""          $GLCP              # gyro-only gradient penalty 0.01 (d6b's, which calmed the robot)
br k4_high   16e6 "$HIGH"                        # reference that lifts 1.4x higher
br o1_hold4  24e6 "$HOLD4"                       # open loop between lift-off and touchdown events
br k6_combo  16e6 "$SPEED $COM $HIGH" $GLCP      # all of the above + centre of mass range
br o3_hold4g 24e6 "$HOLD4 obs_hold_keep_g=true"  # open loop, gravity live
br k3_com    16e6 "$COM"                         # centre of mass 0.5 cm back to 2.5 cm forward
br o2_hold2  20e6 "$HOLD2"                       # open loop for a whole step (sampled at lift-offs only)
br k5_high2  16e6 "$HIGH2"                       # reference that lifts 2.1x higher
echo "$(date) queue done"
