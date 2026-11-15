#!/bin/bash
# Round 12 queue (~40 min of GPU while the operator is out).
cd "$(dirname "$0")/.."
source scripts/robot12_common.sh
while ps -eo args | grep -qE "^[^ ]*python[^ ]* scripts/train.py"; do sleep 30; done
br q1_comy      16e6 "$COMY $COM"          # sideways (+-2 cm) and fore-aft (-0.5..+2.5 cm) mass ranges
br q2_comy_wide 14e6 "$COMY $COM $WIDE"    # + steps no narrower than 10 cm
echo "$(date) queue done"
