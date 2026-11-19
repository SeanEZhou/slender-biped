#!/bin/bash
# round 23 stage 5 (17:10). g0b (g0 + 20M) declines like the faster cadences (421 -> 318 steps per walk at 10M, the
# tilt penalty growing -0.99 -> -1.25 per step), while z0's lineage (gate on, falls cost 20) improved over its last
# 20M (502 -> 542). Every run since e1 trained with the gate off, and since f1 with falls costing 300. Test the recipe:
# the no-stop sweep (c_D0x75) with z0's recipe (gate at its lift-offs, falls cost 20), from z0.
cd "$(dirname "$0")/.."; source scripts/robot23_common.sh
while pgrep -f "^/bin/bash scripts/robot23_queue5.sh" > /dev/null; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
dg() { out=results/robot23/falls_$1.txt; shift; $G "$@" --n 48 --falls 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" >> $out; }
L() { name=$1; T=$2; shift 2; b23 $name "$@" && ev23 $name $T && dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); }
INIT=runs/z0_y2c/params_0020M.pkl
L g0g_gate  1.2   20e6 "$LANDV $D0" land_v=-100.0 $LATE
echo "$(date) queue7 done"
