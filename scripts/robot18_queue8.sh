#!/bin/bash
# round 18 eighth queue (replaces robot18_queue7.sh's lanes; its extra evals finish first): from n11.
cd "$(dirname "$0")/.."; source scripts/robot18_common.sh
until ! pgrep -f "^/bin/bash scripts/robot18_queue7.sh" >/dev/null && ! pgrep -f "^scripts/../.venv/bin/python scripts/gait_eval.py" >/dev/null; do sleep 20; done
L() { b18 "$@" && ev18 $1; }
INIT=runs/n11_s16r/params_0020M.pkl
L n14_s19   20e6 "$LANDV $S19"  land_v=-100.0
L n15_s20   20e6 "$LANDV $S20"  land_v=-100.0
L n16_s21   20e6 "$LANDV $S21"  land_v=-100.0
echo "$(date) queue8 done"
