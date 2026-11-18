# 2026-11-17 round 21 (walking_clean.md §41). Robot 11-17: p4 at --vx 0.03 is the walker (42 mm strides, 17 mm/s, no
# falls in 78 steps); q7 / q2 shifted their weight late and fell. The operator: a continuous gait as fast as k5 (k5's
# step size, l9 / p4's stability); p4's shift may be too much; a small balance loss must not grow into a fall.
# The tables so far move the body only during each swing (still 60 % of the cycle, surges up to 5x the average):
# c_G* (shift_gait.py --glide) keep p4's table (swing, shift timing, held transfer) with a constant pelvis speed.
source scripts/robot20_common.sh
INIT=runs/p4_gate/params_0020M.pkl
F=hw_ank3_b150_m2_10dof
G13='ref_table="c_G13" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.015,0.04]'
G10='ref_table="c_G10" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.015,0.04]'
G08='ref_table="c_G08" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.015,0.04]'
SHOVE='push_dv=[0.0,0.06] push_dur=[0.08,0.2] push_lat=0.7 push_interval_s=[2.0,4.0]'   # small, mostly sideways shoves
b21() { b17 "$@"; }
ev21() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"
         { for v in 0.015 0.025 0.035; do $G $name --n 48 --vx $v; done
           $G $name --n 48 --vx 0.025 --sim loose; $G $name --n 48 --vx 0.025 --seed 3 --set 'prof_acc=[6.0,60.0,25.0]'
           scripts/gpu_python.sh scripts/push_eval.py walk $name --no-sup --n 16 --dv 0 0.1 --vx 0.025; } 2>&1 \
           | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot21/eval_$name.txt; echo "$(date) evaluated $name"; }
FG07='ref_table="c_FG07" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.015,0.05]'   # leg v2: glide, shift 0.07 (= 0.08 on today's leg's length)
evq() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"      # quick: the shift timing at 0.025 m/s
        $G $name --n 48 --vx 0.025 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot21/evalq_$name.txt; }
