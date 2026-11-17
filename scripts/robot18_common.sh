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
# round 18b: n1 (c_W1, the periodic sway) fell 60 % in the robot sim: the sway heads back toward the swing foot before
# it lands. The same swing with the whole transfer in double support (quintic, through the centre at speed) and the
# shift held from lift-off to 0.05 s past touchdown:
W5='ref_table="c_W5" gait_hz=0.4166667 settle_gate_u=[0.125,0.625]'      # 1.2 s steps, 0.6 s swing
W6='ref_table="c_W6" gait_hz=0.5 settle_gate_u=[0.1133,0.6133]'          # 1.0 s steps, 0.55 s swing
# round 18c: n2 (c_W2, periodic sway at 1 step/s) 4 % up. The minimal fix of l9's own table: the release after
# touchdown ends with the step (release_t 0.35 -> 0.2), so the hip-roll targets no longer jump 4.4 deg (c_S12fix).
S12='ref_table="c_S12fix" settle_gate_u=[0.1667,0.6667]'
up_robot() { awk '/sim robot vx 0.015/ {f=1; next} f && / up / {gsub("%","",$2); print $2; exit}' results/robot18/eval_$1.txt; }
# round 18d: n5 (double-support sway, held through the swing) also 4 % up, falling toward the swing foot at lift-off and
# late swing, like n1 / n2. Separate the swing from the structure: l9's fixed table with the IK straight-down swing
# (c_S13ik), and the walk structure (double-support sway, constant pelvis speed) with l9's bump swing (c_W7bump).
S13='ref_table="c_S13ik" settle_gate_u=[0.1667,0.6667]'
W7='ref_table="c_W7bump" gait_hz=0.4166667 settle_gate_u=[0.125,0.625]'
# round 18e: n7 (l9 + the release fix only) is the round's best so far: robot sim 92 % up (l9 75), loose 75 % (l9 36).
# The same structure at 1 step/s (1.0 s steps, 0.55 s swing), from n7:
S14='ref_table="c_S14fast" gait_hz=0.5 settle_gate_u=[0.1523,0.6523]'
# round 18f: n8 (c_S12fix with the IK straight-down swing) 29 % up: the straight-down swing is what breaks the walk
# tables. l9's bump reaches ~5 cm ahead and retracts ~6 cm as it lowers (swing-leg retraction; the retraction near the
# floor is the rug skim). c_S16ikr: the IK swing with a 6 cm reach-and-retract done while high (by 0.8 of the swing),
# then straight down. From n7.
S16='ref_table="c_S16ikr" settle_gate_u=[0.1667,0.6667]'
# round 18g: n10 (n7's table at 1.0 s steps) 15 % up. Smaller steps from n7: 1.1 s steps (c_S17t11), and no centre
# pause (settle0 0, the shift over 0.4 s: c_S18ns).
S17='ref_table="c_S17t11" gait_hz=0.4545454545 settle_gate_u=[0.1602,0.6602]'
S18='ref_table="c_S18ns" settle_gate_u=[0.1667,0.6667]'
# round 18h: n11 (c_S16ikr: retraction while high, then straight down; from n7) is the round's most robust (robot 90 %,
# loose 85 %, shoves 87 / 84 %), skims no more (lowest point late in the swing 28 / 15 / 6 mm), the gate holds less
# (3.7 %), but lands harder (271 mm/s). From n11: a longer, earlier descent after the retraction (c_S19soft); no centre
# pause (c_S20ns); 1.1 s steps (c_S21t11).
S19='ref_table="c_S19soft" settle_gate_u=[0.1667,0.6667]'
S20='ref_table="c_S20ns" settle_gate_u=[0.1667,0.6667]'
S21='ref_table="c_S21t11" gait_hz=0.4545454545 settle_gate_u=[0.1602,0.6602]'
# round 18i: n7's policy walked the 1.0 s table zero-shot (75 % up) but 10M of training on it fell to 6 %. A short
# fine-tune (4M) of n11 on n11's swing at 1.0 s steps (c_S22t10):
S22='ref_table="c_S22t10" gait_hz=0.5 settle_gate_u=[0.1523,0.6523]'
# round 18j: n15 (c_S20ns: n11's swing, no centre pause; from n11) is the most robust policy yet: robot 94 %, loose
# 94 %, shoves 91 / 87 %; lands at 268 mm/s. From n15: the lift 40 -> 30 mm (c_S23low: the drop 25 % slower, knee
# 51 rad/s^2), and 1.0 s steps on its structure (c_S24t10, a short 4M fine-tune).
S23='ref_table="c_S23low" settle_gate_u=[0.1667,0.6667]'
S24='ref_table="c_S24t10" gait_hz=0.5 settle_gate_u=[0.1523,0.6523]'
# round 18k: the two changes that held, together, from n15: no centre pause (n15) at 1.1 s steps (n16): c_S25t11ns.
S25='ref_table="c_S25t11ns" gait_hz=0.4545454545 settle_gate_u=[0.1602,0.6602]'
