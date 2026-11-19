#!/bin/bash
# round 23 stage 4 (16:10). g1 (ref_q 1) collapsed like f4 (371 -> 178 steps per walk): copying the table is not the
# cause. Its falls: 43 of 48 in the first right swing (u 0.24-0.36, touchdown 0.36), 28 forward and 15 left, the right
# foot still up: the body topples before the swinging foot lands. g0 (no stop at the centre, 1.2 s steps) holds
# (437 -> 422 at 10M). So: the sweep trains, the cadence doesn't. Next: the 1.0 s table at k5's speeds (shorter steps,
# the same body speed: cadence or forward speed?), and g0 + 20M (the robot candidate, as z0 was y2 + 20M).
cd "$(dirname "$0")/.."; source scripts/robot23_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name g0_ctrl" > /dev/null; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
dg() { out=results/robot23/falls_$1.txt; shift; $G "$@" --n 48 --falls 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" >> $out; }
ev23 g0_ctrl 1.2; dg g0_ctrl g0_ctrl --vx 0.025
NG='settle_gate_u=[]'
TERM=termination=-300.0
L() { name=$1; T=$2; shift 2; b23 $name "$@" && ev23 $name $T && dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); }
INIT=runs/z0_y2c/params_0020M.pkl
# D4 rows reach 0.05; k5's command range keeps the body speed (steps 1.0 / 1.2 of k5's at the same command)
L g6_t10slow 1.2  20e6 "$LANDV $D4 $NG cmd_vx=[0.015,0.04]" land_v=-100.0 $LATE $TERM
INIT=runs/g0_ctrl/params_0020M.pkl
L g0b_ctrl  1.2   20e6 "$LANDV $D0 $NG" land_v=-100.0 $LATE $TERM
echo "$(date) queue5 done"
