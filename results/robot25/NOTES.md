# Round 25: z0 made reliable (2026-11-19 from 12:30, simulation; GPU budget 7 h)

Scripts: `scripts/robot25_common.sh`, `robot25_queue*.sh`. Per run: `eval_<run>.txt` (3 speeds, loose hips, slow servos,
shoves 0 / 0.1 / 0.2 m/s), `falls_<run>.txt`, `open_<run>.txt` (opening steps after a held start of 2.0 and 0.5 s).

## The operator's goal

Stop pursuing the continuous gait for now and make z0 highly reliable. Three faults remain on the robot:
1. **A sketchy opening.** The right foot steps outward without going forward, then a short left step. k5 starts
   forward at once.
2. **Occasional risky left steps.** After one, the next right step can swing in the air, and the recovery turns the
   robot.
3. **No catching** once the gait deviates.

## Robot data after the hardware fixes (11:48–12:05)

| run | trials | steps |
|---|---|---|
| z0_y2c | 7 | 31–37 in 6 of them (37–44 s); one fell forward after 3 |
| k5_high2 | 3 | 8, 9, 18 |
| h4s2_t10 (gate-free) | 3 | 4–7 |

## Foot placement from the encoders through the rigid model (scripts/robot25_analysis/steps25.py)

- **dx** is how far the landing foot ends up ahead of the stance foot; **dy** is the foot spacing.

| | first R | first L | steady R | steady L | spacing |
|---|---|---|---|---|---|
| z0, robot | +12..+22 / 135–140 | **+6..+12 / 106–124** | +38 | **+13..+16** | ~131 |
| k5, robot | +16..+20 / 127 | +25..+28 / 128–131 | +26..+35 | +14..+35 | ~132 |
| z0, sim | +12 / 127 | +24 / 111 | +27 / 133 | +26 / **110** | |
| k5, sim | +12 / 119 | +24 / 122 | +11 / 128 | +23 / 123 | |

- **z0 is lopsided, and it learned this, not the table.** c_K2gr and c_T12HH are exactly mirror-symmetric. In the
  sim, z0 lands its left foot narrow (110 against 133 mm) and holds a lopsided standing pose at the clock's 0 (hip
  roll L +4.0 / R +1.9, against k5's +2.3 / +1.3). On the robot, z0's left steps go only +13..+16 mm forward, against
  +38 for the right steps.
- **The opening.** The runner starts with ~2 s of the policy acting at clock 0 under a zero command (`--ready-act`).
  z0 never trained that: every training episode starts the clock at once.

## The risky left steps (scripts/robot25_analysis/risky25.py; 97 left and 101 right swings over 7 z0 trials)

- **The 8 worst left swings** reached roll excursions of 8–10° (typical 5.2°). At the left lift-off the body was
  already rolling toward the left (the swing side) at 7–18°/s, against a typical 1.6°/s.
- **They landed short and narrow:** −12..+13 mm forward and 98–114 mm spacing (typical +13 / 119).
- **The right step before each one was normal.** So the body bounces back toward the left after the weight has gone
  onto the right foot, and the left foot lifts into that bounce.
- **Correlation of the excursion** with pitch at the lift-off is −0.57 (a more upright body is worse) and with the
  roll rate −0.40.
- **The gate never held for them.** It holds only above 0.3 rad/s = 17°/s, and it has no direction.
- **At the gate's own moment (u 0.708)** the rate predicts weakly: half the worst steps were still slow there and
  sped up in the next 40 ms. A direction-aware gate alone wouldn't catch most of them, so this round trains
  robustness instead.

## Stage 1 (12:30): one change at a time from z0, then combined

| run | change |
|---|---|
| z5_hold | start_hold_s [0, 2.5]: the runner's held start (new EnvConfig field) |
| z5_mir | mirror_aug: half the episodes mirrored, exact symmetry |
| z5_shv | small, mostly sideways shoves while walking (round 21's SHOVE: up to 0.06 m/s) |
| z5_hms | all three |

- **Baseline:** z0_y2c under the same evaluation.
- z0 with a 2 s held start in the sim: 94 % up (97 % without).

### z5_hold (13:17): worse

- **Training:** 551 → 540 → 503 steps per walk.

| | z0 | z5_hold |
|---|---|---|
| up at 0.015 / 0.025 / 0.035 | 90 / 94 / 94 % | 85 / 83 / 77 % |
| loose hips | 38 % | **0 %** |
| slow servos | 65 % | 50 % |
| shoves, all up (0 / 0.1 / 0.2 m/s) | 98 / 95 / 61 % | 86 / 75 / 39 % |

- **Opening:** the standing pose flipped (hip roll L +1.7 / R +4.1). The first R step is shorter (+6 mm), and the
  left steps are wider (120 against 111).
- So the held start in training costs general stability. Not a robot candidate.

### z5_mir (13:49): symmetric, but much less stable

- **Training:** 537 → 482 → 454 steps per walk.
- **Up at 0.015 / 0.025 / 0.035:** 52 / 62 / 60 % (z0: 90 / 94 / 94 %).
- **Symmetry worked:** the standing pose is L +2.2 / R +2.5. Both feet land 121–132 mm apart (z0: L 111, R 133), and
  the first L step goes +27 mm forward.
- **But the policy has to relearn half its behaviour, and 20M steps isn't enough.**
- **Pattern:** every stage-1 change so far degrades during training (z5_shv 520 → 495 at 10M as well). The control
  run, z5_ctl (queue2), tests whether continuing z0's training degrades by itself.

### z5_shv (14:22): about z0, no better at catching

- **Training:** 520 → 495 → 533.
- **Up at 0.015 / 0.025 / 0.035:** 90 / 92 / 94 % (z0: 90 / 94 / 94 %).
- **Loose hips:** 44 % (z0: 38 %).
- **Shoves, all up:** 95 / 95 / **44 %** at 0 / 0.1 / 0.2 m/s (z0: 98 / 95 / 61 %). Shoves up to 0.06 m/s in
  training don't teach recovery from a 0.2 m/s shove.
- **Standing pose** more lopsided (L 0 / R +4.1).

### z5_hms (14:56): hold + mirror + shoves

- **Training:** 571 → 530 → 626, the highest training score of the round. But training includes the held start,
  mirrored episodes and shoves, so it isn't comparable with the others.
- **Up at 0.015 / 0.025 / 0.035:** 92 / 77 / 67 %. Loose hips 40 %, slow servos 65 %.
- **Opening:** the standing pose is lopsided the other way (L 0 / R +4.1). The first R step barely goes forward
  (+5 mm), the first L step +22. Left landings 123 mm apart.
- **Mirroring with the held start didn't make it symmetric.** In a mirrored episode the held pose is mirrored too,
  but z5_hms still stands lopsided. Not better than z0.

### z5_ctl (15:30): the control, z0 continued 20M with nothing changed, degrades

- **Up at 0.015 / 0.025 / 0.035:** 88 / 85 / 75 % (z0: 90 / 94 / 94 %). Loose hips 17 % (z0: 38 %), slow servos 54 %
  (65 %). Shoves at 0.1 / 0.2 m/s: 77 / 56 % (95 / 61 %).
- **So 20M more on z0's recipe loses robustness by itself.** z0 is a good checkpoint, and further training drifts.
- **Against this control, the shoves help.** z5_shv holds z0's level: 90 / 92 / 94 %, loose 44 %, shoves at 0.1 m/s
  95 %. z5_hold and z5_mir lose more than the control.

### z5_mir2 (16:11): z5_mir +30M, recovers partly

- **Up at 0.015 / 0.025 / 0.035:** 77 / 73 / 67 % (z5_mir 52 / 62 / 60, z0 90 / 94 / 94). Loose hips 8 %, slow
  servos 54 %. Shoves at 0.1 / 0.2 m/s: 77 / 41 %.
- **Still symmetric:** the standing pose is L +2.5 / R +3.3, the first R step goes +18 mm forward, the left landings
  are 121–124 mm apart.
- **Training:** noisy (469 / 455 / 541 / 463).
- **Not a robot candidate yet.** Symmetry costs too much robustness at this training length.

### z5_shv10 (16:45): shoves up to 0.10 m/s, from z0. The best normal walking of the round

- **Up at 0.015 / 0.025 / 0.035:** **96 / 96 / 98 %** (z0: 90 / 94 / 94 %). Shoves at 0.1 / 0.2 m/s: 95 / 59 % (z0
  95 / 61).
- **Weaker:** loose hips 15 % (z0 38 %), slow servos 54 % (65 %).
- **Opening:** the standing pose is lopsided like z0's (L +4.8 / R +2.0). The first R step goes +9 mm forward; the
  left landings are 125 mm apart (z0 111).

### z5_shh (17:17): shoves + sideways mass offset ±6 mm + start tilt 0.08. Worse

- **Training:** 462 → 380 → 370.
- **Up at 0.015 / 0.025 / 0.035:** 73 / 71 / 79 %. Loose hips 6 %, slow servos 29 %. Shoves at 0.1 / 0.2 m/s: 69 / 45 %.
- **The harder states cost more than they teach in 20M.** Not a candidate.

### z5_shv2 (17:50): z5_shv +20M with shoves to 0.10 m/s. The steadiest

- **Up at 0.015 / 0.025 / 0.035:** **98 / 98 / 98 %** (z0 90 / 94 / 94). One fall in 48 in the falls run (z0: 3, all
  left).
- **Loose hips:** **52 %** (z0 38). Slow servos 46 % (65).
- **Shoves at 0.1 m/s:** 100 % (z0 95); at 0.2 m/s 53 % (61).
- **Slower:** 28 / 30 / 33 cm in 20 s against z0's 35 / 39 / 41. Shorter steps; it trades distance for steadiness.
- **Opening:** the standing pose is nearly level (L +2.4 / R +3.4). The first R step barely goes forward (+5 mm),
  the first L step +19. The left landings are 129–131 mm apart (z0 111): the narrow left step is gone.

### z5_shv15 (18:23): shoves to 0.15 m/s. Faster but less steady

- **Up at 0.015 / 0.025 / 0.035:** 85 / 88 / 88 %, walking 42–49 cm (z0 35–41). Loose hips 25 %, slow servos 44 %.
  Shoves at 0.1 / 0.2 m/s: 84 / 42 %.
- **Training:** 481 → 433 → 424.
- **Opening:** the first R step goes +14 mm forward, the first L +29. But the left landings are narrow again (109 mm).

### The risky-step pattern in the sim (vx 0.02, 32 walkers × 20 s; scratchpad risky_sim.py)

- **Measure:** per swing, the roll rate toward the swing side at the lift-off and the roll excursion toward the swing
  side over the next 0.8 s. Normal z0 steps lean about 7° that way; the robot's risky ones reached 8–10°.

| run | L lift-offs rolling toward the swing side > 7°/s | L excursion p50 / p99 | L swings > 8° | R swings > 8° |
|---|---|---|---|---|
| z0_y2c | 26 % | 7.2 / 10.5 | 24 % | 34 % |
| z5_shv10 | 34 % | 7.5 / 10.5 | 35 % | 51 % |
| z5_shv2 | 12 % | **9.1 / 15.0** | **80 %** | 43 % |

- **None of the candidates reduces the risky left-swing pattern.** z5_shv2 lifts off calmer but leans further toward
  the left during the swing. Its "steadiness" (98 % up) comes with larger lean on left swings, so watch its left steps
  on the robot.
- **Up in this run:** z0 94 %, z5_shv10 97 %, z5_shv2 88 %. That is 4 of 32 down, against 98 % in gait_eval's 48 per
  speed, so the up figures are noisy at this sample size.

## Summary (budget 12:30–19:30; last run ended 18:23)

1. **No change fixed the three faults outright.**
   - **The held start** (the runner's 2 s at clock 0 in training) cost general stability and didn't fix the opening.
     In the sim, the first R step is short for every policy (+5..+15 mm).
   - **Mirroring** made z0 symmetric (level stance, both feet ~125 mm apart) but cost 30 points of up-rate. 50M
     recovered only part of it (67–77 %).
   - **Harder states** (mass offset, start tilt) cost more than they taught.
2. **The control matters.** z0 trained 20M more on its own recipe degrades (88 / 85 / 75 %, loose 17 %). z0 is a good
   checkpoint, and further training drifts.
3. **Shoves are the one change that beats the drift.**
   - z5_shv10 (shoves to 0.10 m/s, from z0): 96–98 % up, z0's distance minus about 10 %.
   - z5_shv2 (z5_shv +20M at 0.10): 98 % at all speeds, loose hips 52 % (z0 38), every 0.1 m/s shove survived. But it
     is about 20 % slower and leans further on left swings.
4. **The opening and the risky left steps need something other than training randomization.** For example:
   - a start that shifts the weight before the first lift;
   - a direction-aware gate tested on the robot;
   - a reference with a slightly larger left-side margin.

## For the robot (from hardware/leg_p0/software, all `--enable --act-lpf 0.7 --video`, cable lifted)

| run | command | sim | watch for |
|---|---|---|---|
| z0_y2c | `--run z0_y2c --vx 0.02` | 90–94 % up | the baseline |
| z5_shv10 | `--run z5_shv10 --vx 0.02` | 96–98 %, near z0's steps | first R step short |
| z5_shv2 | `--run z5_shv2 --vx 0.02` | 98 %, shove-proof at 0.1 m/s, wider left landings, shorter steps | larger lean on left swings |

## Robot trials (2026-11-19 21:25-22:01, cable lifted, `--vx 0.02 --act-lpf 0.7`)

Operator: the new policies didn't work well; they learned to wait longer and take more conservative steps.

| trial | how it ended | steps | gate held | first steps R / L / R (dx/dy mm) | note |
|---|---|---|---|---|---|
| z0 212558 | stopped, 26.6 s | 22 | 0.2 s | +17/134, +12/121, +49/142 | |
| z0 220052 | fell forward, 3.7 s | 3 | 0 | +19/141, +7/103, +43/145 | first left step narrow |
| z0 220108 | fell forward, 3.8 s | 3 | 0 | +21/141, +8/106, +37/148 | first left step narrow |
| z0 220125 | fell forward, 3.8 s | 3 | 0 | +20/140, +9/106, +41/141 | first left step narrow |
| z0 220139 | fell left, 37.7 s | 30 | 1.2 s | +23/143, +12/111, +47/146 | fell during a left swing |
| shv10 215409 | fell backward, 13.9 s | 10 | 1.3 s | +19/149, +30/128, +45/153 | |
| shv10 215434 | fell left, 2.9 s | 2 | 0.3 s | +16/152, +13/105 | first left step narrow |
| shv10 215525 | stopped, 37.7 s | 28 | 3.5 s | +17/149, +28/133, +37/149 | |
| shv2 215805 | fell right, 11.1 s | 7 | 1.9 s | +18/151, +9/131, +23/151 | |
| shv2 215842 | fell left, 26.1 s | 18 | 4.2 s | +13/150, +9/128, +26/148 | fell after a left lift-off rolling left at 11.5°/s |

- **The opening decides 4 of 10 trials.** Every trial whose first left step landed 103-106 mm from the right foot fell
  on the next step (3 of 5 z0 trials, forward, at 3.7 s); every one at 111 mm or more got through.
- **z5_shv2 waits.** Its clock was held 1.9-4.2 s per trial (z0 0-1.2 s) and it took 18 steps in 26 s (z0 22 in 26.6).
  Its left landings are wider (128-136 mm, as in the simulator), but it still fell over a left step that lifted off
  while the body rolled left.
- **The falls are still single bad steps the policy doesn't correct.** Left swings: roll excursion p50 5.4°, worst
  9.7-11.7° (2 falls); the worst ones lifted off while already rolling left (-5 to -12°/s), as in round 25's analysis.
- In the simulator z0 stays up 94 % of the time, but only **25 %** with per-step execution errors (round 26's
  `step_err` 0.5 / 2.0° hip roll / 2.9° hip pitch): one sub-par step and it falls, as on the robot.

The operator's diagnosis (round 26): the policy only knows how to follow the gait. From a bad state (already leaning
left before a right step) it still shifts its weight as far as the reference says and tips. It needs training in bad
states and a reason to recover to a good posture instead of following the reference. Shoves (round 25) only made it
slower and more conservative; the goal is a long walk undisturbed, not push resistance.
