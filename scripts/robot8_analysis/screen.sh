#!/bin/bash
# Round 8 screen: per run (run[:params] [act_lpf]), eval6 on the ring fit, the ring fit + pelvis mass 1.5 cm forward,
# the round-6 fit + 1.5 cm, and the hard ring; 6 x 30 s each. Output: results/robot8/screen_<run>.txt
cd "$(dirname "$0")/../.."
mkdir -p results/robot8
run=$1; lpf=${2:-}
X=(); [ -n "$lpf" ] && X=(act_lpf=$lpf)
out=results/robot8/screen_${run//:/_}${lpf:+_lpf$lpf}.txt; : > $out
for s in "r7" "r7 com_dx=0.015" "fitted com_dx=0.015" "r7hard"; do
  set -- $s
  EVAL_PROCS=3 .venv/bin/python scripts/eval6.py $run --sim $1 --n 6 --set "${X[@]}" ${@:2} 2>&1 | grep -v warp | sed "s/\[$1\]/[$s]/" >> $out
done
cat $out
