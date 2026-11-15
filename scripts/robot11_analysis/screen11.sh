#!/bin/bash
# Round 11 screen (2026-11-14): per run (run[:params] [act_lpf] [vx]), eval6 6 x 30 s on h4's simulator + pelvis mass
# 1.5 cm forward, the loose hip roll + 1.5 cm, the round-7 ring + 1.5 cm, and the hard ring; then the static gains on h4's 188 s robot log.
# act_lpf default: the run's own if trained with one, else 0.6 (as on the robot). vx default 0.02.
# Output: results/robot11/screen_<run>[_lpfX][_vxY].txt
cd "$(dirname "$0")/../.."
mkdir -p results/robot11
run=$1; lpf=${2:-}; vx=${3:-0.02}; r=${run%%:*}
if [ -z "$lpf" ]; then
  lpf=$(.venv/bin/python -c "import json;print(json.load(open('runs/$r/config.json'))['env'].get('act_lpf',0.0) or 0.6)")
fi
out=results/robot11/screen_${run//:/_}_lpf${lpf}_vx${vx}.txt; : > $out
for s in "fitted com_dx=0.015" "loose com_dx=0.015" "r7 com_dx=0.015" "r7hard"; do
  set -- $s
  EVAL_PROCS=3 .venv/bin/python scripts/eval6.py $run --sim $1 --n 6 --vx $vx --set act_lpf=$lpf ${@:2} 2>&1 | grep -v warp | sed "s/\[$1\]/[$s]/" >> $out
done
PYTHONPATH=scripts/robot7_analysis .venv/bin/python scripts/robot8_analysis/gains.py $r 2>&1 | grep -v warp >> $out
cat $out
