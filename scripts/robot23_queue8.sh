#!/bin/bash
# round 23 stage 6 (17:50). g0g (the no-stop sweep with z0's recipe: gate on, falls cost 20) improved in training
# (428 -> 502 steps per walk, tilt penalty -0.83 -> -0.59), where the same table with the gate off and falls at 300
# declined (g0b 421 -> 272). Training needs the gate; at faster cadences the gate's 0.3 rad/s stall holds the clock
# 40-66 % of the time (d1, d3). Last two: 1.1 s steps with the gate on, and 1.0 s with a relaxed gate (0.6 rad/s,
# at most 0.3 s), both from g0g.
cd "$(dirname "$0")/.."; source scripts/robot23_common.sh
while pgrep -f "^/bin/bash scripts/robot23_queue7.sh" > /dev/null; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
dg() { out=results/robot23/falls_$1.txt; shift; $G "$@" --n 48 --falls 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" >> $out; }
L() { name=$1; T=$2; shift 2; b23 $name "$@" && ev23 $name $T && dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); }
D7G='ref_table="c_D7t11" gait_hz=0.4545455 settle_gate_u=[0.1477,0.6477] cmd_vx=[0.016,0.044]'
D4R="$D4 settle_gate_w=0.6 settle_gate_max=0.3"
INIT=runs/g0g_gate/params_0020M.pkl
L g7g_t11   1.1   20e6 "$LANDV $D7G" land_v=-100.0 $LATE
L g4r_t10   1.0   20e6 "$LANDV $D4R" land_v=-100.0 $LATE
echo "$(date) queue8 done"
