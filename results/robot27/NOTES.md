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
