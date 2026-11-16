# 2026-11-16 round 18 (walking_clean.md §38): continuous walking. Robot round 17: l9_s18 (c_S9mid18) steps softest,
# but its swing hangs a flat board just above the rug for the last third (the "trip"), the release after touchdown runs
# past the step and jumps the hip-roll targets 3-4 deg at every step boundary, and the settle gate stops the clock 8 %
# of the time. New tables (shift_gait.py walk): one smooth periodic sway, constant pelvis speed, quintic lift and a
# vertical set-down by IK, no holds. From l9_s18 (20M) on its recipe; gait_hz and the gate lift-offs follow each table.
source scripts/robot17_common.sh
INIT=runs/l9_s18/params_0020M.pkl
W1='ref_table="c_W1" gait_hz=0.4166667 settle_gate_u=[0.125,0.625]'      # 1.2 s steps, 0.6 s swing (l9's cadence)
W2='ref_table="c_W2" gait_hz=0.5 settle_gate_u=[0.1133,0.6133]'          # 1.0 s steps, 0.55 s swing
W3='ref_table="c_W3" gait_hz=0.4166667 settle_gate_u=[0.1055,0.6055]'    # 1.2 s steps, 0.7 s swing, 45 mm
W4='ref_table="c_W4" gait_hz=0.5 settle_gate_u=[0.1016,0.6016]'          # 1.0 s steps, 0.6 s swing, flatter sway
b18() { b17 "$@"; }
ev18() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"; P="scripts/gpu_python.sh scripts/push_eval.py walk"
         { $G $name --n 48 --profile; $G $name --n 48 --sim loose; $G $name --n 32 --vx 0.03; $P $name --no-sup --n 16 --dv 0 0.1; $P $name --no-sup --n 16 --dv 0 0.1 --sim loose; } 2>&1 \
           | grep -vi "warning\|^ *warnings\.\|^W[0-9]" > results/robot18/eval_$name.txt; echo "$(date) evaluated $name"; }
