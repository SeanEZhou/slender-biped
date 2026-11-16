#!/bin/bash
# 2026-11-15: WSL hangs when RAM runs out (7.9 GB). Every 5 s: if MemAvailable < 700 MB, kill the newest train.py
# (the lane started last) before the VM thrashes. Logs to runs/mem_guard.log.
while true; do
  av=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$av" -lt 700 ]; then
    pid=$(ps -eo pid,etimes,cmd --sort=etimes | awk '$3 ~ /python/ && $4 == "scripts/train.py" {print $1; exit}')
    echo "$(date) MemAvailable ${av} MB: killing newest train.py pid $pid" >> runs/mem_guard.log
    [ -n "$pid" ] && kill -9 $pid
    sleep 30
  fi
  sleep 5
done
