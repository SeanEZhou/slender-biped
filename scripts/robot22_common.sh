# 2026-11-18 round 22 (walking_clean.md §42). Robot 11-18 00:07-00:24: x1 (glide, p4's shift) steps on time and
# smoothly but falls after ~9 steps (10 of 12 trials over the left foot: the sideways rock grows ~1 deg per cycle);
# x2 is the most reliable but no different from l9; x3 (shoves) twitches and falls forward on step 3. The operator:
# p4's hip sway is too much and costs speed; go back to k5 (its step size and speed are a hard requirement), l9 the
# benchmark; make k5's steps continuous and stable through the reference gait: 1. a continuous pelvis (glide) on k5's
# steps, 2. landing timing, 3. hip-sway timing with some magnitude, less than p4's, and never a longer swing than k5's
# 0.45 s (lift-off to touchdown).
# Tables (shift_gait.py step, k5's c_T12HH parameters: 1.2 s steps, 0.45 s swing, lift 0.75 / 0.75, lift-off at
# 0.5 s, so the settle gate stays at u 0.2083 / 0.7083):
#   c_K1g     + glide                                   (legs apart at lift-off 13.7 deg, at touchdown 3.7)
#   c_K2gr    + glide, shift held to touchdown (rel0 1)  (13.8, 13.7)
#   c_K3gs10  + glide, rel0 1, shift 0.10 complete 0.1 s before lift-off (--shift-lead 0.1)  (11.5, 11.4)
#   c_K4gs08  as K3 with shift 0.08                      (9.2, 9.2)
#   c_K5gs10p K3 + foot over its spot by 80 % of the swing, then straight down (--place-frac 0.8 --rise-frac 0.4)
# p4's c_P1s13: 14.9 at lift-off and touchdown, 0.55 s swing; l9's c_S9mid18 20.6, 0.6 s.
source scripts/robot17_common.sh
INIT=runs/k5_high2/params_0016M.pkl
F=hw_ank3_b150_m2_10dof
K1='ref_table="c_K1g"'; K2='ref_table="c_K2gr"'; K3='ref_table="c_K3gs10"'; K4='ref_table="c_K4gs08"'; K5='ref_table="c_K5gs10p"'
LATE=late_shift=-2000.0
b22() { b17 "$@"; }
ev22() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"
         { for v in 0.015 0.025 0.035; do $G $name --n 48 --vx $v; done
           $G $name --n 48 --vx 0.025 --sim loose; $G $name --n 48 --vx 0.025 --seed 3 --set 'prof_acc=[6.0,60.0,25.0]'
           scripts/gpu_python.sh scripts/push_eval.py walk $name --no-sup --n 16 --dv 0 0.1 --vx 0.025; } 2>&1 \
           | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot22/eval_$name.txt; echo "$(date) evaluated $name"; }
# stage 2 (from y2_k2, the best of stage 1: k5's shift held to touchdown + glide): reference-gait changes on c_K2gr
#   c_Z1soft  rise over 35 % of the swing, lower over 65 % (was 45 / 55): a slower descent within the same 0.45 s
#   c_Z2lead  k5's 0.12 shift complete 0.1 s before lift-off (--shift-lead 0.1)
#   c_Z3t11   1.1 s steps (0.40 s shift, same 0.45 s swing, 0.2 s release): +9 % cadence; same stride at vx x 1.09
#   c_Z4r08   the release starts at 80 % of the swing (the robot's hip roll trails 120-140 ms: lands as it starts)
Z1='ref_table="c_Z1soft"'; Z2='ref_table="c_Z2lead"'; Z4='ref_table="c_Z4r08"'
Z3='ref_table="c_Z3t11" gait_hz=0.4545455 settle_gate_u=[0.2045,0.7045]'
# stage 3: every z fine-tune from y2 came out worse than y2 (z1 71-77 % up, z4 23-40, z3 33-44), so new tables train
# from k5 like stage 1, and y2's table gets a second seed.
#   c_K6r115  c_K2gr with the release starting 0.07 s after touchdown (rel0 1.15; z4's early release fell)
K6='ref_table="c_K6r115"'
