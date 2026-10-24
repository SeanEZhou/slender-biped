#!/bin/bash
# Health watchdog for unattended GPU queues. Exits (so the caller is notified) on:
#   queue runner dead | nvidia-smi failing | GPU idle >= 5 min with work queued | newest train log stale >= 15 min
# and otherwise after HEARTBEAT_MIN minutes with a status line.
cd "$(dirname "$0")/.."
HEARTBEAT_MIN=${HEARTBEAT_MIN:-30}
idle=0
for i in $(seq 1 $((HEARTBEAT_MIN * 2))); do
  sleep 30
  if ! pgrep -f queue_runner.sh >/dev/null; then echo "ALERT queue runner not running"; exit 1; fi
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || { echo "ALERT nvidia-smi failed"; exit 1; }
  queued=$(grep -v '^\s*#' runs/queue.txt | grep -c -v '^\s*$')
  if [ "${util:-0}" -lt 5 ] && { [ "$queued" -gt 0 ] || pgrep -f "scripts/train.py" >/dev/null; }; then idle=$((idle+1)); else idle=0; fi
  if [ $idle -ge 10 ]; then echo "ALERT GPU idle 5 min (queued=$queued)"; exit 1; fi
  log=$(ls -t runs/*.log 2>/dev/null | grep -v queue_runner | head -1)
  if pgrep -f "scripts/train.py" >/dev/null && [ -n "$log" ]; then
    age=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
    if [ $age -ge 15 ]; then echo "ALERT $log stale ${age} min"; exit 1; fi
  fi
done
log=$(ls -t runs/*.log | grep -v queue_runner | head -1)
echo "OK heartbeat $(date +%H:%M) gpu=${util}% queued=$queued running=$(basename $log .log)"; tail -1 "$log"
