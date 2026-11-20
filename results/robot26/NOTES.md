# Round 26 pilot: recovery from bad states (2026-11-19 22:16-23:34)

Goal (operator): a long undisturbed walk. The policy only follows the gait. From a bad state (already leaning left
before a right step) it still shifts as far as the reference says and tips. It must learn to step from a sub-par
position and get back to a good posture. Pushes are not the goal (round 25's shoves made it wait).

New in `mjx_env.py`:
- `step_err=(prob, hip_roll_std, hip_pitch_std)`: at each reference lift-off, with probability prob, random servo-target
  errors on the hip rolls and hip pitches, unseen by the policy, held for one step. The simulator's steps then go
  wrong the way the robot's do.
- `ref_relax=(tilt0, tilt_scale, w0, w_scale)`: ref_q, ref_contact and late_shift fade once the body tilts past tilt0 or
  rotates faster than w0.

z0 under step_err [0.5, 0.035, 0.05] stays up 25 % (94 % without): the robot's "one bad step and it's done".

Bug found: gait_eval / push_eval / opening.py load the run's own config, so the first z6_er evaluations ran with the
errors on. They now default to `step_err=()`, and z6_er was re-evaluated.

## Results (fine-tunes from z0_y2c, z0's recipe; vx 0.025 unless noted; 48 walkers x 20 s)

| run | change | up at 0.015 / 0.025 / 0.035 | errors 0.5/2.0°/2.9° | errors 0.6/2.9°/4.0° | loose hips |
|---|---|---|---|---|---|
| z0_y2c | | 90 / 94 / 94 % | 25 % | | 38 % |
| z6_er (15M) | errors + relaxed reference (8°, 0.8 rad/s) | 71 / 77 / 79 % | 15 % | 2 % | 2 % |
| z6_e (15M) | errors only | 92 / 92 / 92 % | 29 % | 2 % | 23 % |
| z6_es (10M) | smaller errors (1.1° / 1.7°) + relaxed reference | 77 / 73 / 73 % | 23 % | 4 % | 2 % |

Opening (held start 2 s): unchanged everywhere. The first right step goes +7 to +12 mm and the first left step lands
103-111 mm from the right foot (z0 111 mm). On the robot, a first left step at 103-106 mm fell every time.

## Conclusions

- **None learned to recover.** In 10-15M, training with the errors barely moved the score under the errors
  (z6_e 29 % vs 25 %). The training episodes stayed about as short (z6_er 307 -> 318 steps of 1000).
- **Relaxing the reference alone hurts.** Both relaxed runs lost 15-20 points of normal walking and collapsed with
  loose hips (2 %). With nothing else telling it how to recover, the policy just tracks the gait less well.
- The policy may be missing the inputs to recover: it sees only the current moment and the clock. SuperMimic
  (arXiv 2610.28480) / BeyondMimic give the policy 10 past observations and 10 future reference frames. They use a
  very forgiving tracking reward (0.3 m / 0.4 rad Gaussians, no per-event penalties), no pushes, and restarts weighted
  toward where the policy does badly.

## Candidates for the overnight round

1. Observation history (obs_hist; the robot runner has to stack the same frames) + small step errors, trained
   longer (60M+), from z0, or from scratch on z0's reference.
2. Reference preview in the inputs (the next few reference frames instead of only the clock phase).
3. Restarts from the robot's bad states: the narrow first left step, a lift-off while rolling toward the swing side.
4. A forgiving reward: late_shift and the -100 penalties scaled down everywhere, wider ref_q.

No robot candidate from the pilot. z6_e is the only one as steady as z0 in normal walking, and it's not clearly better.

## Robot trials (2026-11-20 00:48-00:52, cable lifted, `--vx 0.02 --act-lpf 0.7`)

Operator: z6_e is not better than z0. Its start is even sketchier (the feet bang together; sometimes it stays up and
then walks like z0). After a step that leaves it leaning left, it seems to take the next right step sooner than z0.

| trial | how it ended | first steps R / L (dx/dy mm) |
|---|---|---|
| z6_e 004826 | stopped 34.4 s | +6/136, +12/87 |
| z6_e 004923 | fell forward 3.9 s | +8/144, +11/85 |
| z6_e 004937 | fell forward 4.1 s | +5/137, +15/86 |
| z6_e 004954 | fell right 37.9 s | +1/132, +16/107 |
| z6_e 005157 | fell left 2.9 s | +5/137, +9/79 |
| z6_e 005210 | fell left 34.6 s | +8/138, +9/80 |
| z0 005049 | stopped 38.9 s | +21/138, +11/119 |
| z0 005142 | fell forward 4.1 s | +17/136, +8/102 |

z6_e's first left step lands 79-87 mm from the right foot (z0 102-119): the feet nearly touch. See round 27's notes for
why the start goes wrong.
