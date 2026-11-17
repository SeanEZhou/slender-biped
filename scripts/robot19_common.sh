# 2026-11-16 evening, round 19 (walking_clean.md §39). Robot 18:52-19:01: n15 / n18 / n19 / n7 fall within 3-6
# steps (l9: 19 touchdowns per fall). The operator:
#   1. the step is "done" mid-air: the robot already leans before the foot is down;
#   2. far too much lean (n15's legs open ~30 deg at lift-off against l9's ~20);
#   3. the leaning makes steps longer and shorter (the gate held n15's clock 14-31 % of the time on the robot).
# On the robot the joints trail their commands by 80 ms (knee, hip pitch) and 120-140 ms (hip roll), so a release
# that starts at the scheduled touchdown starts with the foot still in the air.
# Tables: hold the shift 0.1 s past the scheduled touchdown, then release over 0.2 s (rel0 1.18, release_t 0.3), no
# centre pause, a 0.35 s shift, a 0.55 s swing; shift 0.13 (from 0.18).
source scripts/robot18_common.sh
P1='ref_table="c_P1s13" settle_gate_u=[0.1484,0.6484]'     # n15's swing, shift 0.13, release held 0.1 s
P2='ref_table="c_P2s18" settle_gate_u=[0.1484,0.6484]'     # n15's swing, shift 0.18, release held 0.1 s
P3='ref_table="c_P3bump" settle_gate_u=[0.1484,0.6484]'    # l9's swing, shift 0.13, release held 0.1 s
ev19() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"
         { $G $name --n 48 --profile; $G $name --n 48 --sim loose; $G $name --n 48 --seed 3 --set 'prof_acc=[6.0,60.0,25.0]'; } 2>&1 \
           | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot19/eval_$name.txt; echo "$(date) evaluated $name"; }
GATE58='settle_gate_att=[0.087,0.140]'      # round 19: the attitude gate's limits 3 / 6 -> 5 / 8 deg
