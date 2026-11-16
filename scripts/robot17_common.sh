# 2026-11-16 round 17 (walking_clean.md §37): smoother, softer steps. The robot's gyro jolt is 3-4x mid-stance in the
# 0.1 s after every touchdown (k5, m1, m2, m2b alike), and the simulator lands feet at ~260 mm/s. c_T12HH's design:
# the weight shift unwinds from lift-off (the body falls onto the swinging foot) and a 0.45 s swing asks the servos for
# more acceleration than they have. Gaits from slender_biped/shift_gait.py (--set-frac/--hover/--place-frac/--rel0),
# the soft-touchdown penalty land_v (downward foot speed below land_h). From m2_attland (round 16's best) on its recipe.
source scripts/robot16_common.sh
INIT=runs/m2_attland/params_0014M.pkl
LANDV='land_h=0.02'
LONGU='settle_gate_u=[0.1667,0.6667]'      # the 0.6 s swing tables lift off at 0.40 s of the 1.2 s step
b17() { name=$1; S=$2; extra=$3; shift 3; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S"
        run $name $F $S "$T12 $W $LEAN $SER $HIGH2 $ATT $extra" --init $INIT --rw $STEP $C3M tilt_lim=-0.05 foot_tilt=-30.0 "$@"
        echo "$(date) end $name rc=$?"; }
ev17() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"; P="scripts/gpu_python.sh scripts/push_eval.py walk"
         { $G $name --n 48; $G $name --n 48 --sim loose; $P $name --no-sup --n 16 --dv 0 0.1; $P $name --no-sup --n 16 --dv 0 0.1 --sim loose; } 2>&1 \
           | grep -vi "warning\|^ *warnings\.\|^W[0-9]" > results/robot17/eval_$name.txt; echo "$(date) evaluated $name"; }
