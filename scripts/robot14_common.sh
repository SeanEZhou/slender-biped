# 2026-11-15 round 14 (walking_clean.md §33): recovery steps by imitation. From k5_high2 on its recipe, with the gait
# reference replaced by the recovery-step library rs1 (slender_biped/rec_step.py) while a step runs: command input =
# (dx, dy outward, side), clock = the step's. Standing only: no gait clock, no settle gate, no heading command.
source scripts/robot10_common.sh
INIT=runs/k5_high2/params_0016M.pkl
REC='rec_table="rs1" cmd_zero_prob=1.0 heading_cmd=false settle_gate_u=[]'
STEPA='rec_rand=1.5 rec_rest_s=[0.6,1.5]'                     # stage A: a random step every ~1.3-2.2 s
CP='rec_cp=true rec_cp_th=0.035 rec_cp_gain=1.2'               # stage B: the capture point triggers steps
SHOVE='push_dur=[0.08,0.2] push_interval_s=[2.0,4.0] push_walk=false'
IMIT='ref_q=6.0 ref_contact=6.0 ref_swing=6.0'                 # imitation as the anchor (k5: 3 / 3 / 6)
OFF='progress=0.0 stride=0.0 track_lin=0.0 track_ang=0.0'       # walking terms that read the command as a speed
br() { name=$1; S=$2; extra=$3; shift 3; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S"; run $name $F $S "$T12 $W $LEAN $SER $HIGH2 $REC $extra" --init $INIT --rw $STEP $C3M tilt_lim=-0.05 $OFF $IMIT "$@"; echo "$(date) end $name rc=$?"; }
