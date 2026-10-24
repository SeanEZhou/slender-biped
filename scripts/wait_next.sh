#!/bin/bash
# Block until runs/queue_done.txt gains a line (a queued run finished) or 100 min pass; print status.
cd "$(dirname "$0")/.."
n0=$(wc -l < runs/queue_done.txt)
for i in $(seq 1 600); do
  sleep 10
  if [ "$(wc -l < runs/queue_done.txt)" -gt "$n0" ]; then tail -n +$((n0+1)) runs/queue_done.txt; tail -3 runs/queue_runner.log; exit 0; fi
done
echo "timeout; runner log:"; tail -3 runs/queue_runner.log; for f in $(ls -t runs/*.log | head -1); do tail -2 $f; done
