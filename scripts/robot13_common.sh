# 2026-11-15 round 13 (fall recovery, walking_clean.md §32): from k5_high2 on its own recipe (robot12_common.sh without
# the mass ranges). Hand-like shoves (push_dv: force on the pelvis, the whole robot gains dv m/s over push_dur s) and the
# walk -> stand -> resume supervisor (sup: the command drops to 0 when the IMU predicts a tip; slender_biped/recovery.py).
source scripts/robot10_common.sh
INIT=runs/k5_high2/params_0016M.pkl
SHOVE='push_dur=[0.08,0.2] push_interval_s=[2.5,5.0]'
STAND='cmd_zero_prob=1.0'
SUP='sup=true sup_roll=0.21 sup_pitch=0.21 sup_tau=0.1'
br() { name=$1; S=$2; extra=$3; shift 3; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S"; run $name $F $S "$T12 $W $LEAN $SER $HIGH2 $SHOVE $extra" --init $INIT --rw $STEP $C3M tilt_lim=-0.05 "$@"; echo "$(date) end $name rc=$?"; }
