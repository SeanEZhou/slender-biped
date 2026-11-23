#!/bin/bash
# round 29 queue (2026-11-22 01:30, GPU until ~09:30). Most important first.
cd "$(dirname "$0")/.."; source scripts/robot29_common.sh
late() { [ $(date +%H%M) -gt 0845 ] && [ $(date +%H%M) -lt 1200 ] && { echo "$(date) past 08:45, not starting $1"; return 0; }; return 1; }
L29 z9_a1m 20e6 "$START" land_v=-100.0 $LATE --anchor 1.0 --mirror 1.0
L29 z9_a5m 20e6 "$START" land_v=-100.0 $LATE --anchor 5.0 --mirror 1.0
echo "$(date) queue done"
