# 2026-11-14 round 11 (walking_clean.md §28-29): from k5_high2 (c_T12HH, the step form the next milestone needs).
# Robot round 10: its sideways falls are lift-offs taken while the body still rolls outward; the fitted sim doesn't
# do that, a looser hip roll (k ~5, the tilt test's 3.5-5) does. The open-loop policies don't fight the sideways sway
# but let a fore-aft rock grow. Sourced by robot11_queue.sh.
source scripts/robot10_common.sh
INIT=runs/k5_high2/params_0016M.pkl
# hip-roll give centred looser: k 7 (soft 1.75 above 0.15 N m); with dr_flex 0.5-1.5: k 3.5-10.5 (fitted 10, 2.5)
SER='hidden_flex=[["hip_roll",7.0,0.15,1.75],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
# roll gate: also hold each lift-off while |roll - slow mean + 0.2 s x roll rate (2 Hz)| > 6 deg
RGATE='settle_gate_roll=0.105'
# sideways open loop: roll inputs and joints held between lift-offs / touchdowns; gravity x/z and a 1 Hz pitch rate
# live; inputs live until the first lift-off (clean engage)
SIDE='obs_hold_u=[0.2083,0.3958,0.7083,0.8958] obs_hold_pitch_lpf=1.0 obs_hold_late=true'
br() { name=$1; S=$2; extra=$3; shift 3; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S"; run $name $F $S "$T12 $W $LEAN $SER $HIGH2 $extra" --init $INIT --rw $STEP $C3M tilt_lim=-0.05 "$@"; echo "$(date) end $name rc=$?"; }
