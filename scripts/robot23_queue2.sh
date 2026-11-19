#!/bin/bash
# round 23 stage 1b (13:10). d1_half (c_D1h, half the double support) fell in the simulator (2 % up): at 0.825 s steps
# the body never turns slower than the settle gate's 0.3 rad/s, so the gate held the clock 40 % of the time, and 9 of
# 15 falls started at a held lift-off (u 0.10-0.11 / 0.61). The gate puts the wait back between steps. Every faster
# table hits it, so after d3_amb (gate on, running) the tables train with the gate off (settle_gate_u=[]).
# First the diagnostics: z0's shake baseline, d1 without the gate (as trained with it), d3's evaluation and falls.
cd "$(dirname "$0")/.."; source scripts/robot23_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name d3_amb" > /dev/null; do sleep 30; done
echo "$(date) d3_amb training ended"
ev23 d3_amb 0.6
G="scripts/gpu_python.sh scripts/gait_eval.py"
dg() { out=results/robot23/falls_$1.txt; shift; $G "$@" --n 48 --falls 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" >> $out; }
dg z0_y2c z0_y2c --vx 0.025
dg d1_half d1_half --vx 0.0364
dg d1_half d1_half --vx 0.0364 --set 'settle_gate_u=[]'
dg d3_amb d3_amb --vx 0.05
dg d3_amb d3_amb --vx 0.05 --set 'settle_gate_u=[]'
echo "$(date) diagnostics done"
NG='settle_gate_u=[]'
L() { name=$1; T=$2; shift 2; b23 $name "$@" && ev23 $name $T && dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); }
L e1_half   0.825 25e6 "$LANDV $D1 $NG" land_v=-100.0 $LATE
L e3_amb    0.6   25e6 "$LANDV $D3 $NG" land_v=-100.0 $LATE
L e6_ambq   0.6   25e6 "$LANDV $D6 $NG" land_v=-100.0 $LATE
L e2_halfd  0.825 25e6 "$LANDV $D2 $NG" land_v=-100.0 $LATE
echo "$(date) queue2 done"
