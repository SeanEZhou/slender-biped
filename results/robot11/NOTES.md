# Robot round 11 (2026-11-14 19:04-20:19, simulation only): k5 + sideways open loop + roll gate + looser hips

Goal (operator): steps like k5, doesn't fight the safe sideways sway like the o policies, keeps the fore-aft rock in
check, manages the sideways roll. ~90 min of GPU. Settings `scripts/robot11_common.sh`, queues
`scripts/robot11_queue{,2,3}.sh`, screen `scripts/robot11_analysis/screen11.sh` (adds the `loose` sim).

## Before training: what the simulator says about k5's sideways falls

`scripts/robot10_analysis/sim_liftoff.py` (fitted + mass 1.5 cm forward, filter 0.6, vx 0.02; lift-off = the foot 6 mm
up, as in the robot analysis):

| run | sim | roll at lift-off toward stance | roll rate | falls |
|---|---|---|---|---|
| h4_t05 | fitted | +2.5° | +14°/s | 2/6 |
| k5_high2 | fitted | +0.1 | **-8 (already returning)** | 2/6 |
| h4_t05 | hip roll k 5 / 1.25 (2x give) | +2.3 | +13 | 3/6 |
| k5_high2 | hip roll k 5 / 1.25 | +1.1 | **+20 (robot: +21)** | **6/6** |

The fitted simulator does not reproduce the robot's late, still-rolling lift-offs; a hip roll with twice the give
does, and ranks h4 / k5 as the robot does. New eval sim `loose` (eval6.py). Baseline there: k5 6/6 falls in 3.8 s,
h4 4/6.

## Runs (all from k5_high2 at 16M, k5's recipe; "loose" = training hip roll k 7 / 1.75, x dr_flex 0.5-1.5 = k 3.5-10.5)

| run | change | steps | training episode length (start -> end, of 1000) | fitted: falls | loose: falls |
|---|---|---|---|---|---|
| k5_high2 (ref) | | | 468 at 16M | 1/6, 49 cm | 6/6, 3.8 s |
| `p2_side_gate` | sideways open loop (roll + joints held, gravity x/z + 1 Hz pitch rate live, live until the first lift-off) + roll gate 6° + loose | 24M | 128 -> 106 | 6/6 forward in 1.6 s | 6/6 forward in 1.5 s |
| `m2_gate` | roll gate 6° + loose | 16M | 376 -> 425 (reward 2815 -> 3267) | **6/6, 11.7 s** (4/6 with the gate off) | 6/6, 4.8 s |
| `p3_side_fit` | p2 on the fitted hip range | stopped at 8 min (1 Hz problem, below) | 143 | | |
| `p4_side3_gate` | p2 with the pitch rate at 3 Hz | 20M | 127 -> 118 | 6/6 forward in 2.2 s | |

m2 static gains (h4's robot log): knee - hip -0.70°/° of tilt, pitch-rate gain hip 0.053 / knee 0.120 (k5-like).

## Findings

1. **A 1 Hz low-pass on the pitch rate breaks the fore-aft balance**: on k5 alone (no hold, no retraining) it takes the
   fitted sim from 1/6 to 6/6 falls (3 Hz: 1/6). The policy's fast pitch damping, delayed, pumps the pitch. p2 fell
   forward within 1.6 s and never learned out of it.
2. **The sideways hold from k5 + looser hips gives 2.5 s episodes from which PPO doesn't recover in 20-24M** (p2, p4).
   Round 10's o runs (from h4, fitted hips, smaller steps) started at the same length and climbed to 330-510. Too many
   changes at once for the policy to bootstrap from.
3. **The roll gate doesn't hurt** (k5 + gate at run time: 1/6, 50 cm, unchanged) **nor save k5 on loose hips** (6/6).
4. **Training on the looser hip range for 16M made k5 worse everywhere** (m2: 6/6 on fitted, where k5 is 1/6). The loose
   hips are a much harder task; 16M from k5 is not enough. The sim reproduces the robot's failure now; it doesn't yet
   train it away.

Nothing from round 11 is ready for the robot. Worth a robot try on what exists: **k5 with `--gate-roll 6`** (run-time
only, does no harm in simulation; the runner now logs phase / gate_held / roll_pred, which will show whether the robot's
gate leaks).

## Next

- Curriculum, one change at a time: (a) k5 on the loose hip range alone for much longer (40M+), screened on `loose`;
  (b) the sideways hold with the pitch rate raw (no low-pass) or at 3 Hz, from o3 or from k5 on the fitted hips first,
  then the high reference / loose hips.
- Or reach the loose hips gradually (widen dr_flex over training) instead of shifting the whole range at once.

## On the robot: k5 + `--gate-roll 6` (2026-11-14 20:59-21:02, 4 trials; the first logs with phase / gate_held / roll_pred)

Operator: a wild card; some runs very good, some fall after 3-4 steps. Up 11, 27, 22, 8 s; 3 falls, all mid-swing.

- The gate barely acted: 4-7 holds per trial of 0.02-0.22 s, never the 1 s cap. At the clock's lift-off phases
  (0.208 / 0.708) the predicted roll was mostly 0.5-4.5°, under the 6° threshold. So it changed little, and the variance
  is k5's own.
- **But the encoder-FK foot rise comes at clock u ~0.0 / 0.5 (the start of the reference's weight shift), ~0.5 s before
  the reference lift-off the gate checks**, with roll_pred ±6-9° there. Either k5 lifts early on the robot (the
  simulator's true lift-offs are at u 0.21-0.28 / 0.76-0.79, and its encoder FK tracks them within ~0.03), or the
  hip roll's invisible give under the shifting load makes the foot look up while it's down. The logs can't tell; a side
  camera would.
- The falls began during a gate hold at the lift-off phase, |roll_pred| 13-29° by then: the tip had started before the
  clock reached the lift-off.

## On the robot with video: k5, 8 runs (2026-11-14 21:49-21:53; side view 214953-215052, front quarter 215108-215251)

Video: Windows ffmpeg (`policy_stand.py --video`, `video_rec.py`), frames on the log's clock; contact sheets with
`video_frames.py` (videos are git-ignored and never committed). Gate on in the first four, off in the last four.

1. **k5 lifts its foot on time.** In the side view both boards stay flat on the rug through the shift (clock u 0-0.21),
   and the foot leaves at ~u 0.25, just after the clock lift-off. The encoder-FK "rise" at u ~0.0 / 0.5 is not a lift:
   the body rolls 4-5° over the stance leg with both feet down, a give the encoders can't see (hip roll, ankle roll,
   board, rug). So the round-10/11 "lifts off while still rolling" reading was the shift, not the lift; the gate checks
   at the right moment.
2. **The sideways falls start in the weight shift, with both feet down.** Front view, 215108 and 215251: the body
   leans 7.7-8.9° toward the stance side from u 0.03 and keeps going (-11, -15, -23°); in 215251 (gate off) the clock
   was still at u 0.20, held by the settle gate, when it went over. Shift-phase peak roll on the robot: median 6.1°,
   p90 10.4° over 12 runs, growing from 3-4° on the first steps to 6-8°; every fall ended in a shift phase at 18-20°.
   The weight shift overshoots past the stance board's edge.
3. **The roll gate can't see that lean**: it subtracts the roll's slow mean, which had drifted toward the lean
   (215108: roll -7.4°, gate -4.9° at the lift-off).
4. **The side-view fall (215052) was backward**: as the high swing leg kicked forward, the trunk, already 3-5° back,
   pitched to -10 then -20° and tipped over the heels.

Next: the weight shift is the thing to fix (size / speed of the reference shift for the high steps, or a cap on the
lean during double support), not the lift-off timing.
