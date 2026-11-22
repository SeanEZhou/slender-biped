# 2026-11-20 round 28 (walking_clean.md §49): the robot's falls reproduced in the sim, then trained against.
# results/robot28/NOTES.md. scripts/robot28/repro.py puts the sim robot into logged robot states (velocity fitted by
# shooting): from the stuck double supports before the terrible steps the logged policy falls the same way; from
# 0.4 s and more before the onset it doesn't (the left landing tips the robot, not the sim). Fine-tunes from
# z7_anc2 12M (symmetric, the robot's best start) with a share of episodes starting from robot states
# (models/bank/robot28.npz: the bad states of 2/3 of the falls + normal walking states):
#   BANK   reset_bank robot28, 30 % of the episodes
source scripts/robot27_common.sh
mkdir -p results/robot28
INIT=runs/z7_anc2/params_0012M.pkl
ANCW="$ANC ref_anchor_base=false"
A2="$LANDV $IP $CMDV $ANCW $STAG"            # z7_anc2's recipe (with land_v=-100.0 $LATE)
BANK='reset_bank="robot28" reset_bank_prob=0.3'
NG='settle_gate_u=[]'
FLT='warning\|^ *warnings\.\|^W[0-9]\|warp\|nanmean\|nanmedian\|nanpercentile\|return np'
K2RUNS="z7_anc2 z0_y2c z6_e z7_rgq z7_var z5_shv10 z5_shv2 k5_high2"
HOLD=$(.venv/bin/python -c "import json;print(' '.join(json.load(open('results/robot28/holdout.json'))))")
# held-out robot fall states with policy $1 gate-free (results/robot28/hold_<tag>.txt)
evhold() { tag=$(echo $1 | tr ':' '_'); scripts/gpu_python.sh scripts/robot28/repro.py $K2RUNS --policy $1 --files $HOLD --controls 0 \
             --tag hold_$tag 2>&1 | grep -vi "$FLT" > results/robot28/hold_$tag.txt; echo "$(date) held-out falls $1"; }
ev28() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"; SS="scripts/gpu_python.sh scripts/robot27_analysis/startstop.py"
         { for x in 0.015 0.025 0.035; do $G $name --n 48 --vx $x --set "$NG"; done
           $G $name --n 48 --vx 0.025 --sim loose --set "$NG"; $G $name --n 48 --vx 0.025 --set "$NG" "$SERR"
           $SS $name --n 32 --set "$NG"
           scripts/gpu_python.sh scripts/robot27_analysis/limp_sim.py $name --nogate; } 2>&1 | grep -vi "$FLT" > results/robot28/eval_$name.txt
         evhold $name; echo "$(date) evaluated $name"; }
L28() { b27 "$@" && ev28 $1; }
