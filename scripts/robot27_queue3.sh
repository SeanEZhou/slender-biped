#!/bin/bash
# round 27 stage 1b + 2 (03:05-): z7_anc / z7_ancr (anchor in the action's base) fell within 2.4 s from the first
# iteration and never learned (z7_ancr stopped at 13M). The anchor now moves the reward's reference only
# (ref_anchor_base=false). Then from the better of z7_ip / z7_anc2:
#   VAR  step-length variety (ref_v_jitter 0.4: each step's table speed x U(0.6, 1.4))
#   RX1 -> RX2  the reference loosened in two stages; FREE no joint reference; HIST 8 past frames; SERR small step errors
cd "$(dirname "$0")/.."; source scripts/robot27_common.sh
ANCW="$ANC ref_anchor_base=false"
late() { [ $(date +%H%M) -gt 0745 ] && [ $(date +%H%M) -lt 1200 ] && { echo "$(date) past 07:45, not starting $1"; return 0; }; return 1; }
INIT=runs/z0_y2c/params_0020M.pkl
L z7_anc2 25e6 "$LANDV $IP $CMDV $ANCW $STAG"  land_v=-100.0 $LATE
BASE=$(.venv/bin/python scripts/robot27_analysis/pick.py z7_ip z7_anc2); echo "$(date) base $BASE"
REC="$LANDV $IP $CMDV"; [ "$BASE" = z7_anc2 ] && REC="$LANDV $IP $CMDV $ANCW $STAG"
VAR='ref_v_jitter=0.4'
INIT=runs/$BASE/params.pkl
late z7_var  || L z7_var  25e6 "$REC $VAR"                        land_v=-100.0 $LATE
INIT=runs/z7_var/params.pkl
late z7_rx1  || L z7_rx1  25e6 "$REC $VAR ref_sigma_q=0.2"        land_v=-100.0 late_shift=-500.0 ref_q=1.5
late z7_free || L z7_free 25e6 "$REC $VAR"                        land_v=-100.0 late_shift=0.0 ref_q=0.0
INIT=runs/z7_rx1/params.pkl
late z7_rx2  || L z7_rx2  25e6 "$REC $VAR ref_sigma_q=0.4"        land_v=-100.0 late_shift=0.0 ref_q=0.5
INIT=runs/z7_var/params.pkl
late z7_hist || L z7_hist 25e6 "$REC $VAR obs_hist=8"             land_v=-100.0 $LATE
late z7_gq   || L z7_gq   25e6 "$REC $VAR settle_gate_max=0.15"   land_v=-100.0 $LATE   # shorter holds (z7_anc2 waited 21-25 %)
echo "$(date) queue3 done"
# gate-free check (03:-): z7_anc2 held its clock 21-25 % of the walking time (z0 2-4 %). The same policies run without
# the settle gate (as policy_stand --no-gate)
for r in z0_y2c z7_ip z7_anc2 z7_var z7_rx1 z7_free z7_rx2 z7_hist z7_gq; do
  [ -f runs/$r/done.json ] || [ $r = z0_y2c ] || continue
  { for x in 0.025; do scripts/gpu_python.sh scripts/gait_eval.py $r --n 48 --vx $x --set 'settle_gate_u=[]'; done
    scripts/gpu_python.sh scripts/robot27_analysis/startstop.py $r --n 32 --set 'settle_gate_u=[]'; } 2>&1 \
    | grep -vi 'warning\|^ *warnings\.\|^W[0-9]\|warp\|nanmean\|nanmedian\|nanpercentile\|return np' > results/robot27/nogate_$r.txt
done
echo "$(date) gate-free evals done"
# 06:45: one more if time allows: z7_rx1 (the steadiest loosened run) with shorter gate holds
if [ $(date +%H%M) -lt 0745 ]; then
  INIT=runs/z7_rx1/params.pkl
  L z7_rgq 25e6 "$REC $VAR ref_sigma_q=0.2 settle_gate_max=0.15" land_v=-100.0 late_shift=-500.0 ref_q=1.5
  { scripts/gpu_python.sh scripts/gait_eval.py z7_rgq --n 48 --vx 0.025 --set 'settle_gate_u=[]'
    scripts/gpu_python.sh scripts/robot27_analysis/startstop.py z7_rgq --n 32 --set 'settle_gate_u=[]'; } 2>&1 \
    | grep -vi 'warning\|^ *warnings\.\|^W[0-9]\|warp\|nanmean\|nanmedian\|nanpercentile\|return np' > results/robot27/nogate_z7_rgq.txt
fi
echo "$(date) all done"
