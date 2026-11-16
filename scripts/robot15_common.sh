# 2026-11-15 round 15 (walking_clean.md §35): decisive recovery steps (rec_step.build2 -> rs3: one step out up to 8 cm
# fore-aft / 8 cm outward, 5-6 cm lift, little weight shift first so the loaded leg can step out, a hold gate until
# the body is quiet, then the rear / front foot closes up fore-aft and the stepped foot retracts sideways), standing
# and from k5's walking (rec_walk: walk, recover, resume).
source scripts/robot14_common.sh
REC3='rec_table="rs3" cmd_zero_prob=1.0 heading_cmd=false settle_gate_u=[]'
IMIT3='ref_q=6.0 ref_contact=6.0 ref_swing=20.0'           # the big swings paid fully at 3 cm (ref_swing_h)
# standing branch: from a round-14 policy (the command already means dx, dy, side)
bs() { name=$1; S=$2; init=$3; extra=$4; shift 4; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S"
       run $name $F $S "$T12 $W $LEAN $SER $HIGH2 $REC3 ref_swing_h=0.03 $extra" --init $init --rw $STEP $C3M tilt_lim=-0.05 $OFF $IMIT3 "$@"
       echo "$(date) end $name rc=$?"; }
