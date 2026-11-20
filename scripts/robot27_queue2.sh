#!/bin/bash
# round 27 stage 2: from the better of z7_anc / z7_ancr (pick.py), its recipe plus:
#   VAR  step-length variety (ref_v_jitter 0.4: each step's table speed x U(0.6, 1.4))
#   RX1 -> RX2  the reference loosened in two stages; FREE no joint reference; SERR small step errors; HIST 8 past frames
cd "$(dirname "$0")/.."; source scripts/robot27_common.sh
while pgrep -f "^/bin/bash scripts/robot27_queue.sh" > /dev/null; do sleep 30; done
BASE=$(.venv/bin/python scripts/robot27_analysis/pick.py z7_anc z7_ancr); echo "$(date) base $BASE"
A="$ANC"; [ "$BASE" = z7_ancr ] && A="$ANCR"
REC="$LANDV $IP $CMDV $A $STAG"
VAR='ref_v_jitter=0.4'
late() { [ $(date +%H%M) -gt 0745 ] && [ $(date +%H%M) -lt 1200 ] && { echo "$(date) past 07:45, not starting $1"; return 0; }; return 1; }
INIT=runs/$BASE/params.pkl
late z7_var  || L z7_var  25e6 "$REC $VAR"                        land_v=-100.0 $LATE
INIT=runs/z7_var/params.pkl
late z7_rx1  || L z7_rx1  25e6 "$REC $VAR ref_sigma_q=0.2"        land_v=-100.0 late_shift=-500.0 ref_q=1.5
late z7_free || L z7_free 25e6 "$REC $VAR"                        land_v=-100.0 late_shift=0.0 ref_q=0.0
INIT=runs/z7_rx1/params.pkl
late z7_rx2  || L z7_rx2  25e6 "$REC $VAR ref_sigma_q=0.4"        land_v=-100.0 late_shift=0.0 ref_q=0.5
INIT=runs/z7_var/params.pkl
late z7_hist || L z7_hist 25e6 "$REC $VAR obs_hist=8"             land_v=-100.0 $LATE
late z7_err  || L z7_err  25e6 "$REC $VAR $SERR"                  land_v=-100.0 $LATE
echo "$(date) queue2 done"
