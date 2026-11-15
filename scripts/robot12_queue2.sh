#!/bin/bash
# Round 12, part 2 (23:10): q1 (sideways +-2 cm + fore-aft range) came out worse than k5 even with the mass centred
# (fitted, 12 x 30 s: 5/12 vs 2/12). Isolate the sideways range.
cd "$(dirname "$0")/.."
source scripts/robot12_common.sh
while ps -eo args | grep -qE "^/bin/bash scripts/robot12_queue.sh|^[^ ]*python[^ ]* scripts/train.py"; do sleep 20; done
br q3_comy_only 12e6 "$COMY"
echo "$(date) queue2 done"
