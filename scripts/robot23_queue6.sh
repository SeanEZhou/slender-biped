#!/bin/bash
# round 23 stage 5 (17:00). g6 (1.0 s steps at k5's body speeds) collapsed too (394 -> 210 steps per walk): not the
# forward speed. Every cadence faster than 1.2 s collapsed in training (gate on or off, falls cheap or dear, ref_q 3
# or 1, fast or slow), the walks falling in the first 2-3 s; g0 (the no-stop sweep at 1.2 s) trains. Round 18 found
# 1.1 s steps trainable: c_D7t11 is the no-stop sweep at 1.1 s (0.65 s both down in the table), from g0.
cd "$(dirname "$0")/.."; source scripts/robot23_common.sh
while pgrep -f "^/bin/bash scripts/robot23_queue5.sh" > /dev/null; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
dg() { out=results/robot23/falls_$1.txt; shift; $G "$@" --n 48 --falls 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" >> $out; }
NG='settle_gate_u=[]'
TERM=termination=-300.0
D7='ref_table="c_D7t11" gait_hz=0.4545455 cmd_vx=[0.016,0.044]'
L() { name=$1; T=$2; shift 2; b23 $name "$@" && ev23 $name $T && dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); }
INIT=runs/g0_ctrl/params_0020M.pkl
L g7_t11    1.1   20e6 "$LANDV $D7 $NG" land_v=-100.0 $LATE $TERM
echo "$(date) queue6 done"
