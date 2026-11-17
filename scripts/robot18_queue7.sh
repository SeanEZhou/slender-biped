#!/bin/bash
# round 18 seventh queue (after robot18_queue6.sh): quick evals, then smaller steps from n7.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until grep -q "queue6 done" runs/robot18_queue6.out; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
{ for ck in params_0000M.pkl params_0010M.pkl; do $G n10_s14:$ck --n 48; done
  $G n7_s12 --n 48 --set 'settle_gate_u=[]'; $G n7_s12 --n 48 --sim loose --set 'settle_gate_u=[]'; } 2>&1 \
  | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot18/eval_extra1.txt; echo "$(date) evaluated extra1"
L() { b18 "$@" && ev18 $1; }
INIT=runs/n7_s12/params_0020M.pkl
L n12_s17   20e6 "$LANDV $S17"  land_v=-100.0
L n13_s18   20e6 "$LANDV $S18"  land_v=-100.0
echo "$(date) queue7 done"
