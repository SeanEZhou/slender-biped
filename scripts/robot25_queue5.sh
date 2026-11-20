#!/bin/bash
# round 25 stage 5 (written 17:05): z5_shv10 (shoves to 0.10 m/s) walks best (96-98 % up); larger still: to 0.15 m/s.
cd "$(dirname "$0")/.."; source scripts/robot25_common.sh
while pgrep -f "^/bin/bash scripts/robot25_queue4.sh" > /dev/null; do sleep 20; done
[ $(date +%H%M) -gt 1900 ] && { echo "$(date) past 19:00, not starting"; exit; }
SHOVE15='push_dv=[0.0,0.15] push_dur=[0.08,0.2] push_lat=0.7 push_interval_s=[2.0,4.0]'
INIT=runs/z0_y2c/params_0020M.pkl
L z5_shv15 20e6 "$LANDV $K2 $SHOVE15"        land_v=-100.0 $LATE
echo "$(date) queue5 done"
