# 2026-11-19 round 26 pilot (walking_clean.md §47): recovery from bad states. On the robot one sub-par step (a short,
# narrow left landing, a lean toward the swing side) is followed by more of the same: the policy keeps following the
# reference gait from the bad state. Round 25's shoves made it wait longer, not recover. z0 in the simulator under
# per-step execution errors (step_err 0.5 / 2 deg hip roll / 2.9 deg hip pitch): 25 % stay up (94 % without).
#   ERR    step_err: random hip roll / pitch servo-target errors held for one step (the simulator's steps go wrong the
#          way the robot's do), the policy must catch them on the following steps
#   RELAX  ref_relax: the reference terms (ref_q, ref_contact, late_shift) fade when the body tilts past 8 deg or rotates
#          past 0.8 rad/s, so recovering off the reference is not punished (late_shift forced the full shift at
#          every lift-off, even when the body already leaned that way)
source scripts/robot25_common.sh
INIT=runs/z0_y2c/params_0020M.pkl
mkdir -p results/robot26
ERR='step_err=[0.5,0.035,0.05]'
ERR2='step_err=[0.6,0.05,0.07]'
RELAX='ref_relax=[0.14,0.07,0.8,0.5]'
b26() { b17 "$@"; }
ev26() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"; local FLT='warning\|^ *warnings\.\|^W[0-9]\|warp'
         { for x in 0.015 0.025 0.035; do $G $name --n 48 --vx $x; done
           $G $name --n 48 --vx 0.025 --set "$ERR"; $G $name --n 48 --vx 0.025 --seed 5 --set "$ERR2"
           $G $name --n 48 --vx 0.025 --sim loose; } 2>&1 | grep -vi "$FLT" > results/robot26/eval_$name.txt
         scripts/gpu_python.sh scripts/robot25_analysis/opening.py $name --hold 2.0 2>&1 | grep -vi "$FLT" > results/robot26/open_$name.txt
         echo "$(date) evaluated $name"; }
L() { b26 "$@" && ev26 $1; }
