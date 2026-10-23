#!/bin/bash
# Sequential GPU job queue: never runs two trainings at once.
# Each line of runs/queue.txt is the argument list for scripts/train.py (must include --name).
# Lines are consumed top-down; finished lines move to runs/queue_done.txt with a status.
# Watchdog: a run whose log has not grown for STALL_MIN minutes is killed and marked STALLED.
cd "$(dirname "$0")/.."
STALL_MIN=${STALL_MIN:-20}
touch runs/queue.txt runs/queue_done.txt
while true; do
  line=$(grep -v '^\s*#' runs/queue.txt | grep -v '^\s*$' | head -n 1)
  if [ -z "$line" ]; then echo "[queue] empty, waiting"; sleep 60; continue; fi
  # pop the line
  python3 - "$line" <<'EOF'
import sys
p = "runs/queue.txt"; line = sys.argv[1]
lines = open(p).read().splitlines(); lines.remove(line); open(p, "w").write("\n".join(lines) + ("\n" if lines else ""))
EOF
  # never overlap with any other training process (e.g. one started by hand)
  while pgrep -f "scripts/train.py" >/dev/null; do sleep 30; done
  name=$(echo "$line" | sed -E 's/.*--name ([^ ]+).*/\1/')
  log="runs/$name.log"
  echo "[queue] $(date +%H:%M:%S) start $name"
  scripts/gpu_python.sh scripts/train.py $line > "$log" 2>&1 &
  pid=$!
  status=RUNNING
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    age=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
    if [ $age -ge $STALL_MIN ]; then
      echo "[queue] $name stalled ${age} min -> kill"; kill $pid; sleep 10; kill -9 $pid 2>/dev/null; status=STALLED; break
    fi
  done
  wait $pid 2>/dev/null; rc=$?
  if [ "$status" != STALLED ]; then
    if grep -q '"ok": true' "runs/$name/done.json" 2>/dev/null; then status=OK; else status="FAILED(rc=$rc)"; fi
  fi
  echo "$(date +%F_%T) $status $line" >> runs/queue_done.txt
  echo "[queue] $(date +%H:%M:%S) end $name $status"
done
