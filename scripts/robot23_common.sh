# 2026-11-18 round 23 (walking_clean.md §43). Robot round 22: z0_y2c (k5's steps, the shift held to touchdown, a
# continuous pelvis) walks with less lean and shake than k5 and goes on the working list. The operator's next goal is
# continuity: no visible wait between steps, the rear foot lifting as the front one lands. Target metric: the time
# with both feet down, 0.75 s per step on k5 / z0 (of 1.2 s): halve it, and try a near-zero one. Soft landings are a
# stretch goal only. When it fails, find out why (which instability) before tuning.
# Tables (shift_gait.py step --xfer): k5 / z0's swing unchanged (0.45 s, lift 0.75 / 0.75, 30 mm, glide). The side-to-
# side transfer is one continuous sweep centred on the double support, through the centre at speed (k5 / z0 stop the
# body at the centre between the release and the next shift). Same step length as k5 / z0 at the k5-equivalent
# command: the command scales by 1.2 / T_step (faster walking, same steps). The clock starts mid-transfer (level pose).
#   table     T_step  both down  transfer  legs apart at lift-off (z0 13.8 deg)
#   c_D0x75   1.2     0.75       0.75      13.8   control: z0's timing, no stop at the centre
#   c_D4t10   1.0     0.55       0.55      13.8
#   c_D1h     0.825   0.375      0.375     13.8   half the double support (quasi-static)
#   c_D2hd    0.825   0.375      0.55      12.3   half, the sweep running into both swings
#   c_D6a     0.6     0.15       0.30      ~12    ambitious, quick sweep
#   c_D3a     0.6     0.15       0.45       6.9   ambitious, dynamic sway
#   c_D5a     0.55    0.10       0.50       4.4   near zero, dynamic sway
source scripts/robot22_common.sh
INIT=runs/k5_high2/params_0016M.pkl
D0='ref_table="c_D0x75" gait_hz=0.4166667 settle_gate_u=[0.1563,0.6563] cmd_vx=[0.015,0.04]'
D4='ref_table="c_D4t10" gait_hz=0.5 settle_gate_u=[0.1375,0.6375] cmd_vx=[0.018,0.048]'
D1='ref_table="c_D1h" gait_hz=0.6060606 settle_gate_u=[0.1136,0.6136] cmd_vx=[0.022,0.058]'
D2='ref_table="c_D2hd" gait_hz=0.6060606 settle_gate_u=[0.1136,0.6136] cmd_vx=[0.022,0.058]'
D3='ref_table="c_D3a" gait_hz=0.8333333 settle_gate_u=[0.0625,0.5625] cmd_vx=[0.03,0.08]'
D6='ref_table="c_D6a" gait_hz=0.8333333 settle_gate_u=[0.0625,0.5625] cmd_vx=[0.03,0.08]'
D5='ref_table="c_D5a" gait_hz=0.9090909 settle_gate_u=[0.0455,0.5455] cmd_vx=[0.033,0.087]'
b23() { b17 "$@"; }
# ev23 <run> <T_step>: round 22's evaluation at the k5-equivalent commands (0.015 / 0.025 / 0.035 x 1.2 / T_step)
ev23() { name=$1; k=$(python3 -c "print(1.2/$2)"); G="scripts/gpu_python.sh scripts/gait_eval.py"
         v() { python3 -c "print(round($1*$k,4))"; }
         { for x in 0.015 0.025 0.035; do $G $name --n 48 --vx $(v $x); done
           $G $name --n 48 --vx $(v 0.025) --sim loose; $G $name --n 48 --vx $(v 0.025) --seed 3 --set 'prof_acc=[6.0,60.0,25.0]'
           scripts/gpu_python.sh scripts/push_eval.py walk $name --no-sup --n 16 --dv 0 0.1 --vx $(v 0.025); } 2>&1 \
           | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot23/eval_$name.txt; echo "$(date) evaluated $name"; }
