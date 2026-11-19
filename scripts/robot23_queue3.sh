#!/bin/bash
# round 23 stage 2 (14:05). Why d1 / d3 / e1 fall (simulation):
# - not the gate alone: with the gate off (e1, and d1 / d3 evaluated without it) every walk still falls in 2-5 s, in
#   every direction, mostly on one foot, and at low commands too (not the forward speed);
# - training drives it: the walks get SHORTER as training goes on (d1 335 -> 187 steps, d3 221 -> 172, e1 275 -> 106;
#   y2 on z0's table 457 -> 504). Per step, tilt_lim (body past 4.6 deg) costs 4.0 against 0.65 on z0's table, the net
#   reward drops from +7 to +1.8 per step, and a fall costs only 20: falling early is nearly free.
# Fixes: a fall costs 300 (termination=-300), and a cadence curriculum from z0 (1.2 -> 1.0 -> 0.825 s steps) instead
# of k5 straight onto 0.825 s.
cd "$(dirname "$0")/.."; source scripts/robot23_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name e1_half" > /dev/null; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
dg() { out=results/robot23/falls_$1.txt; shift; $G "$@" --n 48 --falls 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" >> $out; }
ev23 e1_half 0.825; dg e1_half e1_half --vx 0.0364
NG='settle_gate_u=[]'
TERM=termination=-300.0
L() { name=$1; T=$2; shift 2; b23 $name "$@" && ev23 $name $T && dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); }
# 1. the reward fix alone, against d1 / e1 (k5 onto c_D1h, gate off)
L f1_half   0.825 25e6 "$LANDV $D1 $NG" land_v=-100.0 $LATE $TERM
# 2. curriculum rung 1: z0 onto 1.0 s steps (c_D4t10), gate off
INIT=runs/z0_y2c/params_0020M.pkl
L f4_t10    1.0   25e6 "$LANDV $D4 $NG" land_v=-100.0 $LATE $TERM
# 3. rung 2: f4 onto half the double support (c_D1h)
INIT=runs/f4_t10/params_0025M.pkl
L f5_half   0.825 25e6 "$LANDV $D1 $NG" land_v=-100.0 $LATE $TERM
# 4. rung 3: f5 onto 0.6 s steps with the quick sweep (c_D6a)
INIT=runs/f5_half/params_0025M.pkl
L f6_amb    0.6   25e6 "$LANDV $D6 $NG" land_v=-100.0 $LATE $TERM
echo "$(date) queue3 done"
