#!/bin/bash
# Evaluate each listed run (stand test, 100 seeds, final params) as soon as its done.json appears.
cd "$(dirname "$0")/.."
for r in "$@"; do
  until [ -f runs/$r/done.json ]; do sleep 30; done
  if grep -q '"ok": true' runs/$r/done.json; then
    .venv/bin/python scripts/evaluate.py $r --test stand --n 100 --procs 6 >/dev/null 2>&1
    .venv/bin/python scripts/show_eval.py ${r}__stand | head -1
  else
    echo "$r did not finish OK"
  fi
done
