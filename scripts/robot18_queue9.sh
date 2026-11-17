#!/bin/bash
# round 18 ninth queue (after robot18_queue8.sh), from n15: a lower lift (softer drop), then 1 step/s by a short fine-tune.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until grep -q "queue8 done" runs/robot18_queue8.out; do sleep 30; done
L() { b18 "$@" && ev18 $1; }
INIT=runs/n15_s20/params_0020M.pkl
L n18_s23   20e6 "$LANDV $S23"  land_v=-100.0
scripts/gpu_python.sh scripts/gait_eval.py n15_s20 --n 48 --set "ref_table=\"c_S24t10\"" gait_hz=0.5 "settle_gate_u=[0.1523,0.6523]" 2>&1 \
  | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot18/eval_n15_on_s24.txt; echo "$(date) evaluated n15 on c_S24t10"
L n17_s24   4e6 "$LANDV $S24"  land_v=-100.0
echo "$(date) queue9 done"
