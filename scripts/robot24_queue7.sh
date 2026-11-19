#!/bin/bash
# round 24 stage 7: the gait start + larger sway (stride tables), and a gentler fine-tune on round 23's 1.0 s table
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
while pgrep -f "^scripts/../.venv/bin/python scripts/train.py --name d4t_t10" > /dev/null; do sleep 30; done
a=$(.venv/bin/python -c "import json; print(int(json.loads(open('runs/d4t_t10/progress.jsonl').readlines()[-1])['eval/episode_r_alive']))")
[ "$a" -ge 200 ] && ev24 d4t_t10 1.0; dg d4t_t10 d4t_t10 --vx 0.03; echo "$(date) d4t_t10 steps per walk $a"
Lq u8s_z0   0.8   20e6 "$LANDV $U8S $NG" land_v=-100.0 $LATE $TL
Lq d4l_t10  1.0   20e6 "$LANDV $D4 $NG" land_v=-100.0 $LATE $TL $LR
Lq s4hs_z0  0.5   20e6 "$LANDV $S4S $NG" land_v=-100.0 $LATE $TL
echo "$(date) queue7 done"
