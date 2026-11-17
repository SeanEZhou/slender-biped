# 2026-11-17 round 20 (walking_clean.md §40). Robot 11-16 23:49-00:03: p1_s13 / p4_gate are the smoothest walkers so
# far (p4: 115 steps without a fall) but slow: they step the commanded 34-36 mm stride at --vx 0.015, and in simulation
# barely lengthen it at vx 0.025-0.03 (the policy flattens the table's stride). The operator: the continuous hip roll
# (l9 on) is the direction; make p4 go farther while keeping its gait and stability.
# Part A, from p4: train at faster commands (tables to 0.06 m/s), pay progress / stride more, 1.1 s steps.
# Part B: the flat-hip leg (horizontal hip servos, longer thigh and shin): models/hw_flathip*, tables rebuilt for it.
source scripts/robot19_common.sh
INIT=runs/p4_gate/params_0020M.pkl
Q1='ref_table="c_Q1v6" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.02,0.06]'                       # p4's table, rows to 0.06 m/s
Q2='ref_table="c_Q2t11" gait_hz=0.4545455 settle_gate_u=[0.1367,0.6367] cmd_vx=[0.02,0.06]'   # 1.1 s steps (0.3 s shift, 0.5 s swing)
b20() { b17 "$@"; }
ev20() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"
         { for v in 0.015 0.03 0.045; do $G $name --n 48 --vx $v; done
           $G $name --n 48 --vx 0.03 --sim loose; $G $name --n 48 --vx 0.03 --seed 3 --set 'prof_acc=[6.0,60.0,25.0]'; } 2>&1 \
           | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot20/eval_$name.txt; echo "$(date) evaluated $name"; }
# Part B: the flat-hip leg (scripts/make_flathip_model.py): thigh 178 / shin 144 mm (today 138 / 111), the hip servos
# in a row under the plate. _bal: the torso + head ~42 mm forward so the robot's com stays where it is today; the plain
# model leaves it 10 mm behind. Tables from the p4 recipe on the new leg (--leg 0.1782,0.17625, shift 0.115 for the
# same sideways pelvis travel), rows to 0.08 m/s.
FB=hw_flathip_bal_10dof; FR=hw_flathip_10dof
F1='ref_table="c_F1v8" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.02,0.06]'
F2='ref_table="c_F2t11" gait_hz=0.4545455 settle_gate_u=[0.1367,0.6367] cmd_vx=[0.02,0.06]'
ev20f() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"
          { for v in 0.015 0.03 0.045 0.06; do $G $name --n 48 --vx $v; done
            $G $name --n 48 --vx 0.045 --sim loose; $G $name --n 48 --vx 0.045 --seed 3 --set 'prof_acc=[6.0,60.0,25.0]'; } 2>&1 \
            | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot20/eval_$name.txt; echo "$(date) evaluated $name"; }
# today's leg with rows to 0.08 m/s (the control for the flat-hip runs at the same commands)
Q3='ref_table="c_Q3v8" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.03,0.08]'
F1F='ref_table="c_F1v8" settle_gate_u=[0.1484,0.6484] cmd_vx=[0.03,0.08]'
evz() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"     # the table alone (zero action)
        { for v in 0.03 0.06; do $G $name --n 48 --vx $v --zero; done; } 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot20/evalzero_$name.txt; }
