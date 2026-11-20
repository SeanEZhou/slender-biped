# 2026-11-19 round 25 (walking_clean.md §46): z0 made reliable. The operator's three faults on the robot (after the
# hardware fixes z0 walks 31-37 steps): (1) a sketchy opening: the right foot steps out without going forward, then a
# short left step; (2) occasional risky left steps (the body rolling toward the left as the left foot lifts, 7-18 deg/s,
# lands short and narrow), after which the next right step can swing in the air; (3) no catching once the gait deviates.
# Findings (results/robot25/NOTES.md): z0 learned a lopsided posture (held start hip rolls L +4 / R +1.9 in the
# simulator; left landings 110 mm apart against 133 on the right; k5 ~120 both), and its training never held the clock at
# 0 as the runner does for 2 s (--ready-act), nor shoved it. Fine-tunes from z0 (z0's recipe), one change at a time,
# then combined:
#   start_hold_s  the runner's held start in training (EnvConfig.start_hold_s, new)
#   mirror_aug    half the episodes mirrored: exact left-right symmetry
#   SHOVE         small, mostly sideways shoves while walking (round 21's)
source scripts/robot22_common.sh
INIT=runs/z0_y2c/params_0020M.pkl
mkdir -p results/robot25
HOLD='start_hold_s=[0.0,2.5]'
MIR='mirror_aug=true'
SHOVE='push_dv=[0.0,0.06] push_dur=[0.08,0.2] push_lat=0.7 push_interval_s=[2.0,4.0]'
SHOVE10='push_dv=[0.0,0.10] push_dur=[0.08,0.2] push_lat=0.7 push_interval_s=[2.0,4.0]'
HARD='dr_com_y=[-0.006,0.006] reset_tilt=0.08'
b25() { b17 "$@"; }
# ev25 <run>: z0's evaluation (0.015 / 0.025 / 0.035, loose hips, slow servos, shoves 0 / 0.1 / 0.2 m/s), the falls,
# and the opening (held start 2 s)
ev25() { name=$1; G="scripts/gpu_python.sh scripts/gait_eval.py"; local FLT='warning\|^ *warnings\.\|^W[0-9]\|warp'
         { for x in 0.015 0.025 0.035; do $G $name --n 48 --vx $x; done
           $G $name --n 48 --vx 0.025 --sim loose; $G $name --n 48 --vx 0.025 --seed 3 --set 'prof_acc=[6.0,60.0,25.0]'
           scripts/gpu_python.sh scripts/push_eval.py walk $name --no-sup --n 16 --dv 0 0.1 0.2 --vx 0.025; } 2>&1 \
           | grep -vi "$FLT" > results/robot25/eval_$name.txt
         $G $name --n 48 --vx 0.025 --falls 2>&1 | grep -vi "$FLT" > results/robot25/falls_$name.txt
         for h in 2.0 0.5; do scripts/gpu_python.sh scripts/robot25_analysis/opening.py $name --hold $h; done 2>&1 \
           | grep -vi "$FLT" > results/robot25/open_$name.txt; echo "$(date) evaluated $name"; }
L() { b25 "$@" && ev25 $1; }
