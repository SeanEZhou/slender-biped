# 2026-11-20 round 27 (walking_clean.md §48): start, stop, and steps from where the feet are. Robot (z0, z6_e): the start
# is the worst part, the feet bang together, and some steps from a normal-looking stance go wild. Findings
# (results/robot27/NOTES.md): the table's zero-speed row stands, so while the command ramps up from 0 the reference is a
# blend of standing and walking: at the first lift-off (vx ~0.005) half the weight shift with half the lift. The
# robot's first right lift-off: hip-roll spread +1.5 deg (later steps -6, the table -13.8): the right foot lifts with
# the weight centred and swings outward; the left step over-corrects (+10..+14) and lands narrow. The simulator does the
# same (z0: +2.7 deg). Fine-tunes from z0 (z0's recipe), in steps:
#   IP     c_K2ip (shift_gait --inplace: the zero-speed row steps in place, full shift and lift) + walk_state (the clock
#          runs only while walking; stops at a rest point once the command is 0) + faster ramp (0.03 m/s^2)
#   CMDV   commands that start and stop: 30 % zero, resampled every 2-5 s, 0.01-0.04 m/s
#   ANC    the reference re-anchored to the legs (hip pitch, knee) at each lift-off: every step goes from where the feet
#          are to the table's landing pose; ANCR also the hip and ankle rolls
#   STAG   episodes start with the feet staggered (the table's double-support pose at +-0.025 m/s, either foot ahead)
#   RX1/2  the reference's grip loosened in stages (ref_q weight, kernel width, late_shift)
#   FREE   no joint reference (ref_q 0, late_shift 0): the contact timing, swing lift, stride and progress terms only
source scripts/robot26_common.sh
INIT=runs/z0_y2c/params_0020M.pkl
mkdir -p results/robot27
IP='ref_table="c_K2ip" walk_state=true cmd_vx_slew=0.03'
CMDV='cmd_vx=[0.01,0.04] cmd_zero_prob=0.3 cmd_resample_s=[2.0,5.0]'
ANC='ref_anchor=["hip_pitch","knee"]'
ANCR='ref_anchor=["hip_pitch","knee","hip_roll","ankle_roll"]'
STAG='reset_stagger=0.025'
SERR='step_err=[0.5,0.02,0.03]'
b27() { b17 "$@"; }
ev27() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"; SS="scripts/gpu_python.sh scripts/robot27_analysis/startstop.py"
         local FLT='warning\|^ *warnings\.\|^W[0-9]\|warp\|nanmean\|nanmedian\|nanpercentile\|return np'
         { for x in 0.015 0.025 0.035; do $G $name --n 48 --vx $x; done
           $G $name --n 48 --vx 0.025 --sim loose; $G $name --n 48 --vx 0.025 --set "$SERR"
           $SS $name --n 32; $SS $name --n 32 --set reset_stagger=0.025; } 2>&1 | grep -vi "$FLT" > results/robot27/eval_$name.txt
         echo "$(date) evaluated $name"; }
L() { b27 "$@" && ev27 $1; }
