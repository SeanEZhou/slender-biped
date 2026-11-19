#!/bin/bash
# round 23 stage 3 (15:20). The curriculum's first rung failed too: f4 (z0 onto 1.0 s steps) 369 -> 232 steps per walk
# while ref_q (copying the table's joint angles) rose 1.38 -> 1.56 per step and tilt_lim (body past 4.6 deg) grew
# -1.21 -> -1.71. z0 untrained on the same table walks 31 % (and on the 1.2 s no-stop sweep, c_D0x75, 69 %). At the
# faster cadence the table played as written doesn't balance, and the copy reward pulls the policy away from balance.
# Tests: ref_q 3 -> 1 (the contact timing, ref_contact, still holds the cadence), and the no-stop sweep at 1.2 s.
cd "$(dirname "$0")/.."; source scripts/robot23_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name f4_t10" > /dev/null; do sleep 30; done
G="scripts/gpu_python.sh scripts/gait_eval.py"
dg() { out=results/robot23/falls_$1.txt; shift; $G "$@" --n 48 --falls 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" >> $out; }
ev23 f4_t10 1.0; dg f4_t10 f4_t10 --vx 0.03
NG='settle_gate_u=[]'
TERM=termination=-300.0
L() { name=$1; T=$2; shift 2; b23 $name "$@" && ev23 $name $T && dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); }
INIT=runs/z0_y2c/params_0020M.pkl
L g1_t10q1  1.0   20e6 "$LANDV $D4 $NG" land_v=-100.0 $LATE $TERM ref_q=1.0
L g0_ctrl   1.2   20e6 "$LANDV $D0 $NG" land_v=-100.0 $LATE $TERM
L g5_halfq1 0.825 20e6 "$LANDV $D1 $NG" land_v=-100.0 $LATE $TERM ref_q=1.0
echo "$(date) queue4 done"
