#!/bin/bash
# Event stream for monitoring: queue start/end lines, progress lines of the active run, errors,
# and a heartbeat warning if no run log has grown for 15 minutes or the queue runner died.
cd "$(dirname "$0")/.."
declare -A seen
qseen=$(wc -l < runs/queue_runner.log 2>/dev/null || echo 0)
for f in runs/*.log; do seen[$f]=$(wc -l < "$f"); done  # only report lines written after start
while true; do
  n=$(wc -l < runs/queue_runner.log 2>/dev/null || echo 0)
  if [ "$n" -gt "$qseen" ]; then sed -n "$((qseen+1)),${n}p" runs/queue_runner.log | grep -v "empty, waiting"; qseen=$n; fi
  for f in runs/*.log; do
    [ "$f" = runs/queue_runner.log ] && continue
    c=$(wc -l < "$f"); p=${seen[$f]:-0}
    if [ "$c" -gt "$p" ]; then
      sed -n "$((p+1)),${c}p" "$f" | grep -E "step |Traceback|Error|Killed|OOM|nan" | grep -v -i warp | sed "s|^|$(basename $f .log): |"
      seen[$f]=$c
    fi
  done
  newest=$(ls -t runs/*.log 2>/dev/null | grep -v queue_runner | head -1)
  if [ -n "$newest" ] && pgrep -f scripts/train.py >/dev/null; then
    age=$(( ( $(date +%s) - $(stat -c %Y "$newest") ) / 60 ))
    [ $age -ge 15 ] && [ $((age % 5)) -eq 0 ] && echo "WARNING: $(basename $newest) silent for ${age} min"
  fi
  pgrep -f queue_runner.sh >/dev/null || { echo "WARNING: queue runner not running"; sleep 300; }
  if ! pgrep -f scripts/train.py >/dev/null && [ -n "$(grep -v '^\s*$' runs/queue.txt)" ]; then :; fi
  if ! pgrep -f scripts/train.py >/dev/null && [ -z "$(grep -v '^\s*$' runs/queue.txt)" ]; then echo "IDLE: GPU idle, queue empty"; sleep 540; fi
  sleep 60
done
