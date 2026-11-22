# Round 27: start, stop, and steps from where the feet are (2026-11-20 01:30-)

Operator (after round 26): the start is the top priority: the right foot steps outward without moving forward, the
left foot comes in (z6_e: the feet bang together), and only then does the gait begin. The wider problem: the robot
only knows the table's feet positions. Risky steps sometimes start from a normal-looking stance, the stance foot slips
on the rug and the body moves toward the stepping leg. Wanted: (1) the next step from any normal foot spacing, including
feet side by side (standing); (2) a graceful stop back to standing. Ideas: a reference that adapts to where the feet
are; a weaker reference with step-size variety.

## Why the start goes wrong (robot logs 11-19 / 11-20, scripts/robot27_analysis/opening_robot.py)

At each lift-off: hip rolls (deg, + = outward) measured, the table's, and the landing.

- **The first right lift-off happens with the weight centred.** The hip-roll spread L - R is about +1.5° (z0: L +5.3 /
  R +3.8), against -6° in later right steps and the table's -13.8°. The right foot lifts with no weight shift, so it
  can only swing outward: first right steps land +1..+8 mm (z6_e) or +17..+21 mm (z0) ahead, 132-144 mm apart.
- **The first left step over-corrects.** Spread +10 to +14° (later left steps +6 to +9), and the left foot lands 74-87
  mm from the right (z6_e) or 102-119 (z0). Every trial that fell at the start fell on the step after that one.
- **Cause: the table's zero-speed row stands.** The command ramps up from 0 (0.01 m/s² in z0's training), so at the
  first lift-off (0.5 s) the table runs at about 0.005 m/s. The table blends its standing row with its walking rows,
  so it asks for half the weight shift (-6.9°) and half the lift (15 mm): a half-step nothing can balance on. Stopping
  passes through the same blend. The simulator shows it too: z0's shift at the first lift-off is +2.7° (table +13.8;
  startstop.py).
- The table's double-support pose also staggers the feet (at 0.02 m/s the left foot is 24 mm ahead at u 0) while the
  robot starts with them side by side.

## What round 27 changes

New in the code:
- `shift_gait --inplace`: the zero-speed row steps in place (full shift and lift, zero stride). Table **c_K2ip** is
  c_K2gr with that row (the walking rows are identical). A ramped command now starts with full-shift steps that only
  grow in length, and stops with steps that shrink to in-place steps.
- `walk_state`: the clock runs only while walking. It starts from its rest point (u 0 or 0.5, both feet down, centred)
  when the command becomes nonzero, and stops at the next rest point once the ramped command is back at 0. The robot
  stands; the episode starts standing (either foot first).
- `ref_anchor`: at each reference lift-off (and at the start of a walk) the reference is re-anchored to the legs'
  measured angles. The offset fades over the leg's next swing, so each step goes from where the feet are to the
  table's landing pose (the operator's adaptive reference, in joint space).
- `reset_stagger`: episodes start with the feet staggered front-back (either foot ahead, up to the 0.025 m/s table's
  30 mm).
- `ref_v_jitter`: each step's table speed varies (training only): step-length variety.
- `policy_stand.py`: walk_state and ref_anchor as trained; keys `p` (stop walking, stand) and `g` (walk again);
  `walking` and `vx_cmd` columns in the log.
- Training commands now start and stop: 30 % zero, resampled every 2-5 s, 0.01-0.04 m/s, ramped at 0.03 m/s².
- New evaluation `startstop.py`: start, stop at 8 s, restart at 13 s; the first steps, the shift at the first
  lift-off, how long the stop takes, the stance while standing.

Literature: Digit's stand/walk controller (van Marum et al. 2024, arXiv 2404.19173) treats standing (both feet down) as
separate from walking in place, as walk_state does. Cassie footstep-constrained walking (Duan et al. 2022) and ALLSTEPS
(Xie et al. 2020) train on randomized step lengths. SuperMimic / BeyondMimic: forgiving tracking rewards, observation
history.

## Runs (fine-tunes from z0_y2c, z0's recipe)

Stage 1: z7_ip (IP + CMDV), z7_anc (+ anchoring hip pitch / knee, staggered starts), z7_ancr (+ the rolls anchored).
Stage 2 from the better anchored run: z7_var (step-length variety), z7_rx1 -> z7_rx2 (the reference loosened),
z7_free (no joint reference), z7_hist (8 past frames), z7_err (small step errors).

## What happened (01:31-07:57)

- **The first anchoring was broken.** z7_anc / z7_ancr put the anchored reference into the action's base, from the
  measured angles. Training episodes lasted ~2.4 s from the first iteration and never improved (z7_anc 0 % up; z7_ancr
  stopped at 13M, kept as runs/z7_ancr_killed). A servo sagging under load moved the base toward the sag at every
  lift-off, and the policy can't see the offset. Fixed with `ref_anchor_base=false`: the anchor moves only the reward's
  reference (z7_anc2 and everything after). The runner anchors only when a run's base was anchored.
- **The anchored runs wait.** The settle gate held their clock 21-33 % of the walking time (z0 2-4 %), so they walked
  20-30 cm in 20 s against z0's 38. Run gate-free (`--no-gate`, the round-24 rule: train gated, run gate-free) they
  walk as far as z0 and stay up as well or better.
- z7_err was replaced by z7_gq (gate holds capped at 0.15 s), and z7_rgq (z7_rx1 + the same cap) was added at the end.

## Results (vx 0.025 unless noted; 48 walkers x 20 s; start/stop test 32 walkers: start, stop at 8 s, restart at 13 s)

Gated as trained:

| run | up 0.015 / 0.025 / 0.035 | loose hips | step errors | walked (0.025) | gate holding | start/stop up at the end | shift at the 1st lift-off (table 13.8) / 1st step dx |
|---|---|---|---|---|---|---|---|
| z0_y2c | 94 / 92 / 96 % | 33 % | 56 % | 38 cm | 4 % | 31 (no stop) | +3.0° / +17 mm |
| z7_ip | 77 / 75 / 79 % | 56 % | 42 % | 51 cm | 5 % | 24 | +12.9° / +37 |
| z7_anc2 | 98 / 98 / 98 % | 85 % | 71 % | 24 cm | 21 % | 31 | +13.1° / +31 |
| z7_var | 94 / 92 / 90 % | 71 % | 75 % | 28 cm | 28 % | 26 | +16.4° / +33 |
| z7_rx1 | 98 / 94 / 96 % | 81 % | 79 % | 22 cm | 27 % | 32 | +14.5° / +31 |
| z7_rx2 | 98 / 88 / 85 % | 54 % | 79 % | 25 cm | 33 % | 31 | +11.0° / +39 |
| z7_free | 94 / 83 / 73 % | 67 % | 40 % | 25 cm | 32 % | 24 | +14.2° / +31 |
| z7_hist | 79 / 81 / 73 % | 8 % | 46 % | 33 cm | 24 % | 27 | +14.1° / +39 |
| z7_gq | 92 / 88 / 71 % | 0 % | 56 % | 44 cm | 1 % | 25 | +12.8° / +30 |
| z7_rgq | 100 / 94 / 90 % | 6 % | 71 % | 41 cm | 3 % | 30 | +11.5° / +34 |

Gate-free (`settle_gate_u=[]`, as `--no-gate` on the robot):

| run | up (0.025) | walked | start/stop up at the end | shift / 1st step dx |
|---|---|---|---|---|
| z0_y2c | 96 % | 38 cm | 32 | +1.1° / +9 mm |
| z7_ip | 77 % | 51 cm | 26 | +10.9° / +35 |
| z7_anc2 | 100 % | 28 cm | 31 | +9.3° / +30 |
| **z7_var** | **96 %** | **39 cm** | **30** | **+14.0° / +34** |
| z7_rx1 | 100 % | 27 cm | 30 | +10.2° / +29 |
| z7_rx2 | 77 % | 33 cm | 29 | +7.6° / +35 |
| z7_free | 90 % | 33 cm | 28 | +11.8° / +31 |
| z7_hist | 79 % | 40 cm | 27 | +12.6° / +45 |
| z7_gq | 85 % | 44 cm | 26 | +11.8° / +32 |
| **z7_rgq** | **96 %** | **42 cm** | **30** | **+10.4° / +37** |

Every walk_state policy stops: the clock stops 1.5-2.4 s after the command goes to 0 (the ramp down plus the wait for a
rest point), and the walkers then stand with the feet 6-13 mm apart front-back and a tilt wobble of 0.1-0.2°.

## Conclusions

- **The start is fixed in the simulator.** Every in-place-table policy shifts its weight fully before the first
  lift-off (+10 to +16° against z0's +1 to +3) and steps forward at once (first step +30 to +45 mm, z0 +9 to +17).
  The first left step is no longer narrow (110-125 mm).
- **Stopping works.** The robot can be told to stop (key `p`) and starts again (`g`) from standing.
- **Reward-only anchoring + staggered starts (z7_anc2) made the steadiest policy of the series** (98 % at all speeds,
  85 % with loose hips, 71 % with step errors, z0 33 / 56), but it learned to wait at the gate.
- **Step-length variety (z7_var) and one stage of loosening (z7_rx1) kept that steadiness.** Loosening further
  (z7_rx2) or dropping the joint reference (z7_free) lost it. History (z7_hist) needs a longer run.
- **Capping the gate holds (z7_gq, z7_rgq) removed the waiting** but lost the loose-hip robustness (0-6 %).

## For the robot (from hardware/leg_p0/software, cable lifted; walk_state: the clock waits until the command ramps up;
## p = stop walking and stand, g = walk again; Enter / q / space disengage as before)

1. `python policy_stand.py run --run z7_var --enable --vx 0.02 --act-lpf 0.7 --video --no-gate` (the best all-round in
   the simulator gate-free: z0's steadiness and pace, full-shift start, stops)
2. `python policy_stand.py run --run z7_rgq --enable --vx 0.02 --act-lpf 0.7 --video` (no waiting as trained, 94 %, 41 cm;
   weak with loose hips)
3. `python policy_stand.py run --run z7_anc2 --enable --vx 0.02 --act-lpf 0.7 --video --no-gate` (the steadiest; slower)

The runner's walk_state and anchoring code is new and untested on hardware: check the robot stands still until the
command ramps, and that after `p` it finishes its step and stands with both feet down.

## Robot trials (2026-11-20 10:04-10:20; scripts/robot27_analysis/limp.py, limp_sim.py)

17 trials, all at vx 0.02 with `--act-lpf 0.7`: z7_var 3 and z7_anc2 8 (both gate-free), z7_rgq 2 and z0 4 (both gated).

**User's observations:**
- The new policies start better, z7_anc2 most of all. Pause (`p`) and resume (`g`) work.
- They walk worse than z0. They limp on the left leg: the left foot lands early and the body leans left, z7_anc2 most
  of all. z7_rgq sometimes works but is mostly unstable.

### Trial outcomes

A forward fall shows in the log as the pitch climbing past +15-24° over about 1 s.

| run | trial | steps | how it ended |
|---|---|---|---|
| z7_var | 100420 | 76 | no fall; 2 stops and restarts |
| z7_var | 101112 | 38 | no fall |
| z7_var | 101312 | 32 | fell left |
| z7_anc2 | 100819 | 34 | no fall |
| z7_anc2 | 100933 | 52 | 2 stops and restarts, then fell left |
| z7_anc2 | 101541 | 34 | no fall |
| z7_anc2 | 101741 | 17 | no fall |
| z7_anc2 | 101412 | 25 | fell left and forward |
| z7_anc2 | 100914, 101501, 101522 | 5 | fell forward on step 5 |
| z7_rgq | 100648 | 11 | fell right |
| z7_rgq | 100719 | 29 | fell left |
| z0 | 101648 | 32 | no fall |
| z0 | 101817 | 38 | fell left |
| z0 | 101936 | 31 | fell left |
| z0 | 101919 | 3 | fell forward |

**The three z7_anc2 step-5 falls happen the same way:**
- step 1 (right) lands +50 mm ahead of the stance foot;
- step 2 (left) lands +31 mm;
- step 3 (right) lands +44 mm;
- step 4 (left) lands -7..-2 mm, 110-114 mm apart, with the body rolling 8° left;
- step 5 (right) lunges +73..+82 mm and lands narrow (106-118 mm); the body pitches forward and falls.

**The start is good.** The first three steps:

| | step 1 (R) | step 2 (L) | step 3 (R) |
|---|---|---|---|
| z7_anc2 | +47..+54 mm | +14..+34 mm | +40..+48 mm |
| z0 | +19..+25 mm | +4..+11 mm | +42..+44 mm |

The trouble starts at step 4, the second left step.

### The left limp: per side, steps 5 on

Weight shift is the hip-roll spread at lift-off toward the stance leg, from the logged command. Landing dx is the swing
foot's distance ahead of the stance foot.

**Robot:**

| run | shift before L / R step | L landing dx / dy (mm) | R landing dx / dy (mm) | L / R time in the air (s, table 0.46) | roll at L touchdown |
|---|---|---|---|---|---|
| z0 | +7.2° / +9.5° | +14 / 120 | +36 / 146 | 0.34 / 0.56 | -3.8° |
| z7_anc2 | **+4.8° / +9.3°** | **+2** / 115 | +41 / 142 | 0.34 / 0.58 | -5.4° |
| z7_var | **+2.4° / +12.5°** | **-7** / 124 | +37 / 153 | 0.32 / 0.54 | -5.9° |
| z7_rgq | +6.8° / +9.9° | +15 / 124 | +24 / 131 | 0.46 / 0.58 | -3.3° |

**Simulator** (gate-free, vx 0.02, 24 walkers):

| run | shift before L / R step | L dx (mm) | R dx (mm) | L / R time in the air (s) |
|---|---|---|---|---|
| z0 | +8.6° / +8.9° | +23 | +27 | 0.44 / 0.44 |
| z7_anc2 (25M) | **+6.0° / +9.2°** | **+12** | +22 | **0.30** / 0.34 |
| z7_var (25M) | **+2.9° / +13.0°** | **+13** | +25 | **0.31** / 0.48 |
| z7_rx1 | +5.9° / +10.1° | +7 | +21 | 0.32 / 0.36 |
| z7_free | +1.7° / +9.5° | +6 | +22 | 0.30 / 0.36 |
| z7_rgq | +3.8° / +9.6° | +15 | +30 | 0.42 / 0.46 |
| z7_ip | +6.8° / +9.7° | +25 | +26 | 0.36 / 0.52 |
| **z7_gq** | **+7.2° / +9.2°** | **+23** | **+24** | **0.46 / 0.48** |
| **z7_anc2 (12M)** | **+8.9° / +9.6°** | **+17** | **+27** | 0.34 / 0.46 |
| z7_var (12M) | +2.7° / +13.5° | +10 | +26 | 0.32 / 0.48 |
| z6_e | +11.6° / +8.0° | +25 | +19 | 0.44 / 0.42 |
| z0 (10M) | +8.9° / +8.1° | +29 | +17 | 0.42 / 0.50 |

- **The limp is the policies', not the robot's.** The simulator shows it too. The table is exactly mirror-symmetric,
  but the round 27 policies learned to shift their weight onto the right leg only weakly before a left step. The body
  then falls onto the left foot early: it is in the air 0.30 s instead of 0.46 s and lands short. On the robot the
  command shows it (L/R +2.4°/+12.5° for z7_var).
- **The robot adds its own left bias of about 20 mm on top.** z0 is symmetric in the simulator (+23/+27 mm) but steps
  +14/+36 mm on the robot.
- **z7_anc2 got lopsided late in training.** At 12M its shift is symmetric (+8.9°/+9.6°); at 25M it is +6.0°/+9.2°.
  z7_var was lopsided from the start (its 12M checkpoint is too). Symmetry drifts as training goes on: z0 is lopsided
  the other way at 10M (+29/+17 mm).
- **z7_rgq is symmetric enough but fragile.** In the simulator it holds up with loose hips only 6 % of the time. On
  the robot it fell after 11 and 29 steps.

### A symmetric checkpoint: z7_anc2 at 12M (results/robot27/nogate_z7_anc2_params_0012M.txt)

Gate-free, simulator:

| | z7_anc2 12M | z7_anc2 25M | z7_gq |
|---|---|---|---|
| up at 0.015 / 0.025 / 0.035 | 96 / 94 / 98 % | 98-100 % | 90 / 81 / 79 % |
| walked | 39 / 43 / 48 cm | 24-28 cm | 37 / 44 / 49 cm |
| up with loose hips | 81 % | 85 % (gated) | 2 % |
| up with step errors | 73 % | 71 % (gated) | 56 % |
| foot clearance p50 / p10 | 27 / 10 mm | 19 / 3 mm | |

Start and stop for the 12M checkpoint:
- first steps +30 / +20 mm (25M: +30 / +14), the first left step 126 mm from the right foot (25M: 112);
- weight shift at the first lift-off +10.4°;
- it stops 1.6 s after the command, the feet then 7 mm apart front-back;
- 28 of 32 walkers are up at the end.

**Next robot candidate:**

```
python policy_stand.py run --run z7_anc2 --params params_0012M.pkl --enable --vx 0.02 --act-lpf 0.7 --video --no-gate
```

### Next: keep the policies symmetric

- **An exactly symmetric policy:** the action is ½ (f(s) + M f(M s)), where M is the mirror image (left-right swap
  plus half a cycle of phase). The runner would run the network twice per step. Literature: Mittal et al. 2024, Su et
  al. 2024.
- **Or:** mirror_aug from a checkpoint that is already symmetric (z7_anc2 12M). Round 25's mirroring started from
  lopsided z0 and lost 30 points.
- **Either way:** a reward for the table's weight shift at every lift-off, both sides.
- **Pick checkpoints by symmetry too** (limp_sim.py), not only by staying up.
