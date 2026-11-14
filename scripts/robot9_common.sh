# 2026-11-14 round 9 (walking_clean.md §25): h4 on a faster, continuous gait (c_T10L: 1.0 s steps, no pause before or
# after the swing, 2-4 cm steps at 0.02-0.04 m/s; robot nominal --vx 0.027 = 1.5x h4's 1.8 cm step), then one branch
# per idea against the 5 Hz shake loop. Sourced by robot9_base.sh / robot9_lane*.sh.
C3M="scuff=-100.0 foot_tilt=-10.0 ang_vel_xy=-1.0 ang_acc=-0.006 action_rate=-0.1 action_acc=-0.1 clear=-2.0"
# mimic the new gait harder than h4 did (ref_q 3 -> 5 with a tighter kernel, tip heights paid, progress 1.5 -> 3,
# stride paid up to the commanded step length)
# 03:40: d1 / d2 (progress 3, ref_swing 6, ref_contact 3) drifted from walking to standing during the fine-tune
# (d1 progress 542 -> 319, d2 580 -> 151): pay walking more from here on
STEP="progress=6.0 stride=50.0 narrow=-1.0 ref_swing=12.0 slip=-100.0 ref_contact=2.0 ref_q=5.0 ref_tip=2.0"
G='ref_swing_h=0.010 ref_table="c_T10L" gait_hz=0.5 settle_gate_u=[0.2,0.7] settle_gate_max=0.2 ref_sigma_q=0.05 stride_cmd=true'
W="$G cmd_vx=[0.02,0.04] narrow_min=0.065 dr_imu_tilt=0.05 dr_com_x=[-0.005,0.025]"
LEAN='hidden_off=[["knee",-0.09,0.0,0.015],["hip_pitch",0.0,0.0,0.015]] drag_back=0.0 drag_back_max=0.0'
SER='hidden_flex=[["hip_roll",10.0,0.15,2.5],["ankle_roll",10.0,0.15,2.5],["knee",15.0],["hip_pitch",15.0],["hip_yaw",10.0]] dr_flex=[0.5,1.5] flex_series=true dr_flex_tc=[0.003,0.008]'
ring() { echo "ring=[[\"pitch\",4.2,0.2,$1,0.02],[\"roll\",5.3,0.25,$2,0.02]] ring_delay=0.02 dr_delay_steps=[2,3]"; }
R05=$(ring 0.5 0.35); R10=$(ring 1.0 0.7); R15=$(ring 1.5 1.05)
F=hw_ank3_b150_m2_10dof
run() { EXTRA_SET="dr_floor_tc=[0.02,0.8] kp_hip_roll=2.5 vy_avg_mode=\"box\" tilt_lim=0.08 $4" scripts/b150_run.sh $1 shift_b150 $2 $3 "${@:5}"; }
BASE_INIT=runs/g1_t10l/params_0040M.pkl
br() { name=$1; extra=$2; shift 2; [ -f runs/$name/done.json ] && return; S=$(cat runs/robot9_steps 2>/dev/null || echo 24e6); echo "$(date) start $name $S"; run $name $F $S "$W $LEAN $SER $extra" --init $BASE_INIT --rw $STEP $C3M tilt_lim=-0.05 "$@"; echo "$(date) end $name rc=$?"; }
