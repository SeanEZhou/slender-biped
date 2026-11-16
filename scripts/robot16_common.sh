# 2026-11-15 round 16 (walking_clean.md §36): k5's robot falls (21:37-21:51, 13 trials, 12 fell) start at a lift-off
# the body isn't ready for: leaning / rolling toward the swing foot (sideways falls) or pitched 13-14 deg forward
# after a right landing (forward falls); the rate-only settle gate held < 0.1 s on 90 % of steps. And the right board
# lands rolled 3.5 deg (49 % beyond 5 deg). From k5 (k5_high2 at 16M) on its own recipe:
source scripts/robot10_common.sh
INIT=runs/k5_high2/params_0016M.pkl
ATT='settle_gate_att=[0.052,0.105]'          # hold a lift-off while leaning > 3 deg toward the STANCE foot (2026-11-16: sides were swapped; behaviour kept) or |pitch| > 6 deg
b16() { name=$1; S=$2; extra=$3; shift 3; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S"
        run $name $F $S "$T12 $W $LEAN $SER $HIGH2 $extra" --init $INIT --rw $STEP $C3M tilt_lim=-0.05 "$@"; echo "$(date) end $name rc=$?"; }
