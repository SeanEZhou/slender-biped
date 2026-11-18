#!/bin/bash
# round 21 third queue (replaces queue 2's w5 / w6). The shift-timing check calibrates against the robot: legs apart
# at lift-off in simulation p4 30 deg (robot 17, works), q2 28 (14, shaky), q7 23 (7.6, fell). w1 (g13) 28, w2 (g10)
# 23, w3 (g08) 17: the smaller shifts lift with less spread than q7. New reward late_shift: at each lift-off, the hip-roll
# spread short of the reference's (rad^2). Runs: the glide gait at 0.13 / 0.10 with it, then with shoves, then leg v2.
cd "$(dirname "$0")/.."; source scripts/robot21_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/gait_eval.py" > /dev/null; do sleep 20; done
ev21 p4_gate
L() { b21 "$@" && ev21 $1; }
LATE=late_shift=-2000.0
INIT=runs/w1_g13/params_0030M.pkl
L x1_g13L  20e6 "$LANDV $G13 $GATE58"          land_v=-100.0 $LATE
INIT=runs/p4_gate/params_0020M.pkl
L x2_g10L  30e6 "$LANDV $G10 $GATE58"          land_v=-100.0 $LATE
INIT=runs/x1_g13L/params_0020M.pkl
L x3_g13Ls 20e6 "$LANDV $G13 $GATE58 $SHOVE"   land_v=-100.0 $LATE
F=$FB; INIT=runs/f1_bal/params_0030M.pkl
L f6_g07   30e6 "$LANDV $FG07 $GATE58"         land_v=-100.0 $LATE
echo "$(date) queue3 done"
