# 2026-11-19 round 24 (walking_clean.md §44). Round 23: the faster cadences of k5 / z0's quasi-static gait (a full
# weight shift, then a step) didn't train, and the settle gate stalls them. The operator: humans and the G1 interleave
# shift and step and barely sway; build a good reference first. New tables from slender_biped/stride_gait.py: constant
# pelvis speed, the LIPM sway about a pivot 12 mm inside the board centre, level stance boards, a joint-space swing.
# The cart-table balance point stays on the boards throughout (kinematic check). The operator picked 3 and 4, with k5's
# 30 mm clearance added (the "h" tables), the originals kept:
#   table      T_step  duty  both down  sway  clearance  peak knee (no-load 516 deg/s)
#   s3_brisk   0.40    0.60  0.08 s     22 mm   6 mm      530
#   s3h_brisk  0.40    0.56  0.04       22     25         467
#   s4_relax   0.50    0.62  0.12       29      8         470
#   s4h_relax  0.50    0.58  0.09       29     30         443
# Commands keep k5's step lengths (18-48 mm): cmd_vx = [0.015, 0.04] x 1.2 / T_step. The settle gate is off (at 0.4-0.5 s
# steps the body never gets below its 0.3 rad/s).
source scripts/robot23_common.sh
INIT=runs/z0_y2c/params_0020M.pkl
NG='settle_gate_u=[]'
S3='ref_table="s3_brisk" gait_hz=1.25 cmd_vx=[0.045,0.12]'
S3H='ref_table="s3h_brisk" gait_hz=1.25 cmd_vx=[0.045,0.12]'
S4='ref_table="s4_relax" gait_hz=1.0 cmd_vx=[0.036,0.096]'
S4H='ref_table="s4h_relax" gait_hz=1.0 cmd_vx=[0.036,0.096]'
b24() { b17 "$@"; }
ev24() { ev23 "$@"; mv results/robot23/eval_$1.txt results/robot24/; }
dg() { out=results/robot24/falls_$1.txt; shift; scripts/gpu_python.sh scripts/gait_eval.py "$@" --n 48 --falls 2>&1 | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" >> $out; }
L() { name=$1; T=$2; shift 2; b24 $name "$@" && ev24 $name $T && dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); }
# stage 2 (00:53): z0 can't jump to 0.5 s steps (s4h_z0: 65 -> 60 steps per walk at 10M; the table open loop: 0 % up,
# the lift ~0.1 s late). A cadence ladder of stride tables, 30 mm clearance (lift-knee 0.9, lift-hip 0.65):
#   u8h_stride   0.80 s steps, duty 0.65, both down 0.24 s, sway 44 mm, knee 302 deg/s
#   u65h_stride  0.65 s, duty 0.62, both down 0.15 s, sway 37, knee 353
# each rung gated (safer: z0's gate at the table's lift-offs) and gate-free (continuous); the operator wants both.
U8='ref_table="u8h_stride" gait_hz=0.625 cmd_vx=[0.0225,0.06]'
U65='ref_table="u65h_stride" gait_hz=0.7692308 cmd_vx=[0.0277,0.0738]'
G8='settle_gate_u=[0.0781,0.5781]'; G65='settle_gate_u=[0.0625,0.5625]'; G4H='settle_gate_u=[0.0469,0.5469]'
# stage 3 (01:25): u8_z0 65 -> 93 steps per walk at 10M: still falling in the first 2 s. The stride tables' sway is a
# pendulum orbit: at u 0 the body crosses the centre at ~0.15 m/s sideways, but every start is from rest with the
# first lift-off 0.1 s in. And x_off -0.03 left the standing pose 0.22 rad off z0's (the centre of mass 20-30 mm ahead
# of the boards: forward falls in the first second). New tables (--ramp-v, --x-off-v, nv 9): sway, lift and the foot
# offset grow linearly with speed from 0 (z0's standing pose) to full at the lowest training command, so the command
# ramp at each start brings the steps in small.
R8='ref_table="u8r_stride" gait_hz=0.625 cmd_vx=[0.0225,0.06]'
R65='ref_table="u65r_stride" gait_hz=0.7692308 cmd_vx=[0.0277,0.0738]'
R4H='ref_table="s4hr_relax" gait_hz=1.0 cmd_vx=[0.036,0.096]'
R3H='ref_table="s3hr_brisk" gait_hz=1.25 cmd_vx=[0.045,0.12]'
G3H='settle_gate_u=[0.0312,0.5312]'
# a failed run (fewer than 200 steps per walk at the end): only the fall diagnosis, not the full evaluation
Lq() { name=$1; T=$2; shift 2; b24 $name "$@" || return 1; a=$(.venv/bin/python -c "import json; print(int(json.loads(open('runs/$name/progress.jsonl').readlines()[-1])['eval/episode_r_alive']))")
       [ "$a" -ge 200 ] && ev24 $name $T; dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); echo "$(date) $name steps per walk $a"; }
# stage 4 (01:57): u8r_z0 130 -> 118 steps per walk (falls at ~2 s, in the first full-size right swings, every
# direction): z0 doesn't adapt to the stride gait even with the start ramp. Its prior (a 1.2 s quasi-static shift,
# 13.8 deg legs apart at lift-off; u8r_z0 spreads them 19 deg mid-swing against the table's few) fights the table.
# Next: from scratch (no --init), the same recipe.
bnew() { name=$1; S=$2; extra=$3; shift 3; [ -f runs/$name/done.json ] && return; echo "$(date) start $name $S (from scratch)"
         run $name $F $S "$T12 $W $LEAN $SER $HIGH2 $ATT $extra" --rw $STEP $C3M tilt_lim=-0.05 foot_tilt=-30.0 "$@"
         echo "$(date) end $name rc=$?"; }
Ln() { name=$1; T=$2; shift 2; bnew $name "$@" || return 1; a=$(.venv/bin/python -c "import json; print(int(json.loads(open('runs/$name/progress.jsonl').readlines()[-1])['eval/episode_r_alive']))")
       [ "$a" -ge 200 ] && ev24 $name $T; dg $name $name --vx $(python3 -c "print(round(0.025*1.2/$T,4))"); echo "$(date) $name steps per walk $a"; }
# stage 5 (02:37): why nothing trains: per step alive, u8r_new pays tilt_lim -11.9 (body past 4.6 deg, deg^2) against
# ~+6 of positive terms; z0 on its own gait pays -0.5. Falling early is the best return, so training learns to fall
# (round 23's collapses had the same signature: tilt_lim 4.0 per step against 0.65). u8rg_z0 (gated): 6 % up, falls at
# the held lift-off: the gate freezes the table mid-sway (the body crossing at speed), so it doesn't suit these gaits.
# Fix: tilt_lim 10x lighter (-0.005), falls cost 100.
TL='tilt_lim=-0.005 termination=-100.0'
# stage 6 (03:01): u8r_t (tilt_lim -0.005): the per-step return is now positive (+5) but the walks don't get longer
# (129 -> 105; in the robot-like evaluation it falls at once). The stride gait isn't stabilised from z0 in 20M. s4hr_t
# stopped. Round 23's continuous-sweep tables (z0's posture and shift, no stop at the centre) collapsed with the gate
# off under the heavy tilt penalty; retried with the light one: gate-free continuous gaits at 1.0 and 0.825 s steps.
# stage 7 (03:08): the start problem (NOTES.md): the pendulum sway needs the body crossing the centre at ~0.15 m/s as a
# foot lifts, which a start from rest can't give. A gait start (EnvConfig.start_u / start_pre; stride_gait.py
# --inplace: the zero-speed row steps in place): the clock starts at mid single support (u 0.25, sideways speed zero)
# and holds 0.8 s while the reference shifts the weight onto the stance foot (0.5 s) and lifts the swing foot (0.3 s).
# And a larger sway: with the table's own (pivot 12 mm inside the board centre, the CoM 22 mm out at mid-stance, the
# board's inner edge at 15.5) the shifted CoM sits on the board's edge and the board tips. Pivot 10 mm outside the
# board centre (--zmp-in -0.01): open loop in the ideal simulator 2.4 s with the start (20 mm outside: 3.2 s).
#   u8s_stride  0.8 s steps, sway 77 mm (both down 0.24 s)   s4hs_relax 0.5 s, sway 51   s3hs_brisk 0.4 s, sway 39
U8S='ref_table="u8s_stride" gait_hz=0.625 cmd_vx=[0.0225,0.06] start_u=0.25'
S4S='ref_table="s4hs_relax" gait_hz=1.0 cmd_vx=[0.036,0.096] start_u=0.25'
S3S='ref_table="s3hs_brisk" gait_hz=1.25 cmd_vx=[0.045,0.12] start_u=0.25'
# stage 7b (03:17): d4t_t10 (c_D4t10, light tilt penalty) 352 -> 205 at 10M: still collapses, so round 23's collapses
# weren't the tilt penalty. A gentler fine-tune: learning rate 5e-5 (default 3e-4), on the same table from z0.
LR='--lr 5e-5'
# stage 8 (04:17): d4l_t10 (lr 5e-5) collapsed faster (357 -> 143 -> 102): not the step size. u8s_z0 (the gait
# start): 103 -> 62. Every gate-off fine-tune degrades, even on a table z0 walks untrained (round 23 g0_ctrl, c_D0x75:
# 71 % untrained -> 54-62 %; g0b on: 421 -> 272 steps), while gated runs improve (g0g). Hypothesis: exploration noise
# (policy std ~0.6, i.e. ~0.15 rad per joint): the gate absorbs the wobble it causes; without it the training walks
# fall early and the policy degrades. Test: z0's params with the noise scale lowered to ~0.25 (runs/z0_lowstd).
TERM=termination=-300.0
# stage 9 (04:40): n0_lowstd (noise ~0.25) 443 -> 376 at 10M, worse than g0_ctrl: not the exploration noise. But the
# gated policies walk without the gate: g0g (c_D0x75) gate off 96 % up (gated 85-90), g7g (c_D7t11, 1.1 s) 52 %.
# Recipe: train with the gate (stable), deploy without it. Ladder from g7g (1.0 s, then 0.825 s), gated, evaluated
# gate-free; and the stride table with the gait start, gated at mid-stance (u 0.25 / 0.75: the sway's sideways speed
# is zero there, the body over the stance foot; the attitude gate off, since the body leans toward the stance foot by design).
GMID='settle_gate_u=[0.25,0.75] settle_gate_att=[]'
# evf <run> <T_step>: the gate-free evaluation (gate off at the three k5-equivalent commands) + its falls
evf() { name=$1; k=$(python3 -c "print(1.2/$2)"); G="scripts/gpu_python.sh scripts/gait_eval.py"
        { for x in 0.015 0.025 0.035; do $G $name --n 48 --vx $(python3 -c "print(round($x*$k,4))") --set 'settle_gate_u=[]'; done
          $G $name --n 48 --vx $(python3 -c "print(round(0.025*$k,4))") --set 'settle_gate_u=[]' --falls; } 2>&1 \
          | grep -vi "warning\|^ *warnings\.\|^W[0-9]\|warp" > results/robot24/free_$name.txt; echo "$(date) gate-free evaluated $name"; }
Lg() { name=$1; T=$2; shift 2; b24 $name "$@" || return 1; evf $name $T; a=$(.venv/bin/python -c "import json; print(int(json.loads(open('runs/$name/progress.jsonl').readlines()[-1])['eval/episode_r_alive']))"); echo "$(date) $name steps per walk $a"; }
D4G='ref_table="c_D4t10" gait_hz=0.5 settle_gate_u=[0.1375,0.6375] cmd_vx=[0.018,0.048]'
D1G='ref_table="c_D1h" gait_hz=0.6060606 settle_gate_u=[0.1136,0.6136] cmd_vx=[0.022,0.058]'
# stage 10 (05:40): h4g gate-free falls (33 of 38) overshoot toward the stance foot: z0's shift (legs 13.8 deg apart
# at lift-off) is too much at 1.0 s steps. u8sg_z0 (stride, gated at mid-stance): 97 -> 69 at 10M. New tables, round
# 23's with the shift 0.12 -> 0.08 (9.2 deg, k5 round 22's c_K4gs08 value): c_D4s08 (1.0 s), c_D1s08 (0.825 s).
D4S='ref_table="c_D4s08" gait_hz=0.5 settle_gate_u=[0.1375,0.6375] cmd_vx=[0.018,0.048]'
D1S='ref_table="c_D1s08" gait_hz=0.6060606 settle_gate_u=[0.1136,0.6136] cmd_vx=[0.022,0.058]'
# stage 11 (06:16): h4s_t10 (shift 0.08, 1.0 s, gated) 332 -> 391 -> 433 in training; gate-free 48 / 42 / 33 % up
# (h4g with z0's shift 0.12: 23 / 19 / 2). More of it (+20M), and a smaller shift still: c_D4s06 (0.06, 6.9 deg).
D4S6='ref_table="c_D4s06" gait_hz=0.5 settle_gate_u=[0.1375,0.6375] cmd_vx=[0.018,0.048]'
# stage 12 (06:57): h1s_half (0.825 s, shift 0.08) gate-free 2-4 % up, falls toward the swinging foot mid-swing (26 of
# 46, u 0.24-0.33): too little shift for that cadence (at 1.0 s, 0.12 overshoots and 0.08 balances). Between them:
# c_D9s09 (0.9 s steps, the shift 0.09 = 10.3 deg), from h4s2.
D9S='ref_table="c_D9s09" gait_hz=0.5555556 settle_gate_u=[0.125,0.625] cmd_vx=[0.02,0.0533]'
