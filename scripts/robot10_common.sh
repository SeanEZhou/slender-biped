# 2026-11-14 round 10 (walking_clean.md §27): single changes from h4_t05 on h4's own recipe (round 6, robot6_queue6.sh),
# their combination, higher-stepping references, and an open-loop-within-a-step policy. Kept from h4: c_T12 timing with
# the settle gate holding up to 1.0 s, no training ring, no action filter in training, the light tilt penalty, the loose
# pose kernel. Sourced by robot10_queue.sh.
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
STEP="progress=1.5 stride=50.0 narrow=-1.0 ref_swing=6.0 slip=-100.0 ref_contact=3.0 ref_q=3.0"
# settle_gate_max / ref_sigma_q: h4's values, set because the defaults have since changed (gate 1.0 -> 0.5)
T12='ref_swing_h=0.010 ref_table="c_T12" gait_hz=0.4166667 settle_gate_u=[0.2083,0.7083] settle_gate_max=1.0 ref_sigma_q=0.1'
W="cmd_vx=[0.015,0.04] stride_target=0.05 narrow_min=0.065 dr_imu_tilt=0.05"
LEAN='hidden_off=[["knee",-0.09,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
F=hw_ank3_b150_m2_10dof
# higher steps: same timing as c_T12, lift_knee = lift_hip 0.60 (H) / 0.75 (HH) instead of 0.45; first-step foot
# clearance open loop at 0.02 m/s: 27.5 / 39 / 59 mm. ref_swing pays fully at a proportionally higher lift.
HIGH='ref_table="c_T12H" ref_swing_h=0.018'
HIGH2='ref_table="c_T12HH" ref_swing_h=0.025'
SPEED='stride_cmd=true'
COM='dr_com_x=[-0.005,0.025]'
GLCP='--lcp 0.01 --lcp-idx 3:6'
# open loop within a step: sensors sampled at the lift-offs (0.2083, 0.7083) and touchdowns (0.3958, 0.8958)
HOLD4='obs_hold_u=[0.2083,0.3958,0.7083,0.8958]'
HOLD2='obs_hold_u=[0.2083,0.7083]'
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
INIT=runs/h4_t05/params_0030M.pkl
# br <name> <steps> "<env overrides>" [extra train.py args]
br() { name=$1; S=$2; extra=$3; shift 3; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S"; run $name $F $S "$T12 $W $LEAN $SER $extra" --init $INIT --rw $STEP $C3M tilt_lim=-0.05 "$@"; echo "$(date) end $name rc=$?"; }
