# 2026-11-22 round 29 (walking_clean.md §50): z0_y2c with start and stop, without losing z0's walking. Round 27's
# start/stop fine-tunes from z0 started well but walked worse on the robot (the left limp: a weak weight shift before
# left steps, learned in training) and round 28 found fine-tunes drift by themselves. New losses (scripts/train.py
# install_anchor_mirror):
#   --anchor C   the mean action pulled toward z0's own while the command is a walking speed (>= 0.02 m/s, fading in
#                from 0.01): standing, starting and stopping are free to change, walking stays z0's
#   --mirror C   left/right symmetry loss on the mean action (Mittal et al. 2024)
# Start/stop recipe as round 27 (in-place table c_K2ip, walk_state, commands that start and stop).
source scripts/robot27_common.sh
INIT=runs/z0_y2c/params_0020M.pkl
mkdir -p results/robot29
ANCW="$ANC ref_anchor_base=false"
NG='settle_gate_u=[]'
FLT='warning\|^ *warnings\.\|^W[0-9]\|warp\|nanmean\|nanmedian\|nanpercentile\|return np'
# ev29 <run>: gated and gate-free walking, loose hips, step errors, start/stop, per-side symmetry (limp_sim)
ev29() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"; SS="scripts/gpu_python.sh scripts/robot27_analysis/startstop.py"
         LS="scripts/gpu_python.sh scripts/robot27_analysis/limp_sim.py"
         { echo "== gated"; for x in 0.015 0.025 0.035; do $G $name --n 48 --vx $x; done
           $G $name --n 48 --vx 0.025 --sim loose; $G $name --n 48 --vx 0.025 --set "$SERR"; $SS $name --n 32
           echo "== gate-free"; for x in 0.015 0.025 0.035; do $G $name --n 48 --vx $x --set "$NG"; done
           $G $name --n 48 --vx 0.025 --sim loose --set "$NG"; $SS $name --n 32 --set "$NG"
           echo "== symmetry"; $LS $name; $LS $name --nogate; } 2>&1 | grep -vi "$FLT" > results/robot29/eval_$name.txt
         echo "$(date) evaluated $name"; }
b29() { b17 "$@"; }
L29() { b29 "$@" && ev29 $1; }
START="$LANDV $IP $CMDV"            # round 27's start/stop recipe (z7_ip)
# turning (round 29): in-place turns as a new target heading (heading_cmd: the policy sees the corrective yaw rate),
# 30 % of the nonzero commands, 29-92 deg; TURNY also the turning reference (hip yaws), at 2/3 of the full rate
TURN='turn_prob=0.3 turn_wz=[0.25,0.8]'
TURNY="$TURN turn_yaw=true turn_yaw_scale=0.67"
evturn() { name=$1; TE="scripts/gpu_python.sh scripts/robot29/turn_eval.py"
           { $TE $name --n 32 --deg 45; $TE $name --n 32 --deg 90; $TE $name --n 32 --deg 90 --set "$NG"; } 2>&1 | grep -vi "$FLT" > results/robot29/turn_$name.txt
           echo "$(date) turn-evaluated $name"; }
L29t() { b29 "$@" && ev29 $1 && evturn $1; }
