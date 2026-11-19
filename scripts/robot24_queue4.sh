#!/bin/bash
# round 24 stage 4: from scratch on the ramped 0.8 s stride table, gate off
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^/bin/bash scripts/robot24_queue3.sh" > /dev/null; do sleep 30; done
Ln u8r_new  0.8   40e6 "$LANDV $R8 $NG" land_v=-100.0 $LATE
echo "$(date) queue4 done"
