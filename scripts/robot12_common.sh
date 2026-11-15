# 2026-11-14 round 12 (walking_clean.md §30): from k5_high2. Robot + video (k5, 16 runs): every fall goes to the robot's
# left; left landings come with the body already 3.3 deg (p10 7) left, growing step by step; right landings +2. In the
# fitted sim a pelvis mass 1-2 cm left reproduces it (k5: 2/6 -> 4/6 falls, all left); k5 trained with only +-5 mm.
source scripts/robot10_common.sh            # round-10 recipe and the fitted hip give (round 11's looser hips didn't train in 16M)
INIT=runs/k5_high2/params_0016M.pkl
COMY='dr_com_y=[-0.02,0.02]'
WIDE='narrow_min=0.10'                      # board centres >= 10 cm apart (was 6.5): don't step toward the centre line
br() { name=$1; S=$2; extra=$3; shift 3; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S"; run $name $F $S "$T12 $W $LEAN $SER $HIGH2 $extra" --init $INIT --rw $STEP $C3M tilt_lim=-0.05 "$@"; echo "$(date) end $name rc=$?"; }
