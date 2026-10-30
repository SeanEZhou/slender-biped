#!/bin/bash
# Round-3 evaluation of one run once it finishes: declared checkpoint rule -> CPU gates (100 seeds) ->
# gait anatomy -> power audit. Usage: scripts/eval_hw.sh <run> stand|walk   (output: results/eval_hw/<run>.txt)
cd "$(dirname "$0")/.."
r=$1; kind=$2
until [ -f runs/$r/done.json ]; do sleep 30; done
mkdir -p results/eval_hw
out=results/eval_hw/$r.txt
if ! grep -q '"ok": true' runs/$r/done.json; then echo "$r FAILED" > $out; cat $out; exit 1; fi
P=$(.venv/bin/python scripts/best_checkpoint.py $r)
echo "run $r  checkpoint $P" > $out
if [ "$kind" = walk ]; then tests="walk40 startstop stand"; else tests="stand push"; fi
for t in $tests; do
  .venv/bin/python scripts/evaluate.py $r --test $t --n 100 --procs 6 --params $P --tag _$P >/dev/null 2>&1
done
.venv/bin/python scripts/qsum.py results/eval/${r}__*_${P}*.json results/eval/${r}__*${P%.pkl}*.json 2>/dev/null | sort -u >> $out
if [ "$kind" = walk ]; then
  .venv/bin/python scripts/gait_anatomy.py $r --params $P --test walk40 --seeds 3 2>&1 | grep -v -i warp | tail -6 >> $out
  .venv/bin/python scripts/power_audit.py $r --params $P --test walk40 --n 20 2>&1 | grep -v -i warp >> $out
fi
.venv/bin/python scripts/power_audit.py $r --params $P --test stand --n 20 2>&1 | grep -v -i warp >> $out
cat $out
