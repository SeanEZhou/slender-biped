# Round 21: a continuous gait, how much weight shift, and small balance losses (2026-11-17 11:36-17:14, simulation)

Goal (the operator, after round 20 on the robot): a continuous gait as fast as k5 (k5's step size, l9 / p4's
stability), the right amount of weight shift (p4's may be too much; the shift is for faster stable steps, excess slows
things down), and a small balance loss that doesn't grow into a fall. Speed has to come from the gait, not from
pushing the policy on a slow gait.

GPU: 5.6 h of 7. Scripts `scripts/robot21_common.sh`, `robot21_queue.sh` .. `robot21_queue4.sh`.

## p4 on the robot: how a small balance loss becomes a fall (11-17 11:16-11:19, 5 trials)

3 of 5 fell (17, 33, 16 s); 2 were stopped early. Per step (`escal.py`-style trace in the scratchpad):
- **The falls are not a slow build-up.** Each comes within one step of a lift-off where the body was 2-4 deg off its
  usual roll for that side.
  - 111602: 4 deg further left than usual at a left lift-off. The policy pushed with the standing hip roll (+7 deg
    over the table, normally +1 to +3), and the body overshot to 14 deg right and fell right.
  - 111908: 3 deg less to the right at a right lift-off. The table's shift kept pushing toward the standing foot,
    and the body went over it to the left.
  - 111632: 2 deg off; it fell toward the lifting side.
- **The table's shift is fixed:** the same amount and timing whatever the body is doing. Only the policy's ±14 deg
  correction can adapt it, and once off it overshoots.
- **On average p4 does correct small errors.** A roll error of >2 deg at lift-off is down to a median 0.9 deg 0.5 s
  later. p1 overcorrects even on average (error x-0.4 half a step later, i.e. rocking). The falls are the tail.

## What round 21 tries

**A continuous gait, `shift_gait.py --glide`.** p4's table (swing, shift timing, the transfer held past touchdown)
with the pelvis at a constant speed: every foot on the floor sweeps back relative to the pelvis through the whole
cycle. The swinging foot blends from the track it lifted off onto the one it lands on. At 0.025 m/s the pelvis moves
at 25 mm/s all cycle (p4's table: 0, then up to 137 mm/s during each swing), and the foot travels ~60 mm per swing
(k5's size).

**The shift amount:** c_G13 (p4's 0.13), c_G115, c_G10, c_G08.

**A small balance loss:** `SHOVE` = small, mostly sideways shoves in training (up to 0.06 m/s every 2-4 s).

**A shift-timing check** (`gait_eval.py`): the legs' hip-roll spread at the real lift-off, and that as a share of the
peak spread in the swing. Calibrated on the robot-tested policies:

| policy | simulator: spread at lift-off, share of peak | robot: spread at lift-off | on the robot |
|---|---|---|---|
| p4 | 30 deg, 91 % | 17 deg | works |
| q2 | 28 deg, 86 % | 14 deg | shaky |
| q7 | 23 deg, 76 % | 7.6 deg | fell on its first steps |

**A training penalty `late_shift`** (`mjx_env.py`): at each real lift-off, the hip-roll spread short of the
reference's (rad^2, weight -2000). Corrections at other times are free.

## Results

All from p4 (x1 / x3 from w1 / x1; f6 from leg v2's f1_bal), 20-30M steps. Robot simulator, 48 x 20 s, act_lpf 0.7;
loose / slow hips at 0.025 m/s; shove test: 0 and 0.1 m/s shoves in four directions while walking at 0.025.

| run | robot up / cm at 0.015 · 0.025 · 0.035 | loose 0.025 | slow hips 0.025 | legs apart at lift-off | landing | shove test up |
|---|---|---|---|---|---|---|
| p4_gate | 90% 33 · 94% 34 · 85% 37 | 73% 32 | 83% 36 | 30.1 | 266 | 95 % |
| w1_g13 | 83% 42 · 83% 41 · 75% 43 | 48% 41 | 73% 39 | 27.9 | 267 | 80 % |
| w2_g10 | 94% 39 · 79% 40 · 81% 42 | 42% 37 | 56% 38 | 22.6 | 265 | 86 % |
| w3_g08 | 96% 41 · 98% 44 · 96% 48 | 58% 43 | 71% 44 | 16.9 | 271 | 96 % |
| w4_g10s | 73% 39 · 79% 43 · 77% 47 | 31% 44 | 65% 44 | 21.7 | 265 | 77 % |
| x1_g13L | 83% 38 · 88% 40 · 79% 44 | 65% 40 | 58% 37 | 29.2 | 256 | 81 % |
| x2_g10L | 100% 35 · 92% 34 · 90% 36 | 79% 35 | 65% 36 | 24.5 | 263 | 93 % |
| x3_g13Ls | 92% 32 · 85% 34 · 85% 39 | 67% 34 | 71% 35 | 27.9 | 272 | 85 % |
| x4_g115L | 65% 36 · 67% 39 · 48% 40 | 31% 35 | 48% 37 | 27.4 | 270 | 61 % |
| f6_g07 | 100% 23 · 96% 27 · 88% 33 | 83% 29 | 92% 27 | 21.0 | 284 | 95 % |

Share of peak spread at lift-off: p4 91 %, x1 89 %, x3 86 %, x4 86 %, x2 79 %, f6 74 %, w4 73 %, w3 67 %.

## What it shows

- **The continuous gait goes farther at p4's shift, and is less robust than p4.**
  - w1 / x1 (c_G13) walk 38-44 cm per 20 s against p4's 33-37.
  - They are upright 75-88 % against p4's 85-94 %, 58-73 % with slow hips against 83 %, and 80 % through shoves
    against 95 %.
  - It is not yet "k5's step size with p4's stability".
- **Less shift in the table becomes a later shift in the policy.** w2 / w3 / w4 lift off with 67-76 % of their peak
  spread, the zone q7 was in. Yet the simulator scores them highly (w3: 96-98 % up, 96 % through shoves). The
  simulator rewards the late shift and the robot punishes it (q7), so the simulator can't pick the shift amount.
  - On today's robot, shifting less than p4 has only been seen to fail or wobble (q7, q2).
  - The lowest that shifts on time in simulation is c_G13 with the penalty (x1, 89 %).
- **The late-shift penalty fixes timing only when the table itself shifts enough.** x1 lifts with 89 % of its peak.
  On c_G10 (x2) the table's own spread is small, so the policy clears it and keeps adding shift after lift-off (79 %).
- **Shoves trade distance for robustness.** x3 (x1 + shoves): 85-92 % up, 71 % slow hips, 85 % through shoves, at
  p4's distance. w4 (c_G10 + shoves) got worse. Whether it stops the robot's one-step escalation is a robot question.
- **Leg v2 on the continuous gait (f6) is the most robust policy of the round:** 88-100 % up, 83 % loose, 92 % slow
  hips, 95 % through shoves. It walks the least (23-33 cm). Its timing check (74 %) is calibrated on today's leg only.
- x4 (c_G115) came out worse than both neighbours (48-67 %): one seed per setting is noisy.

## To test on the robot

All `--enable --act-lpf 0.7 --video`, today's leg.

1. `python policy_stand.py run --run x1_g13L --enable --vx 0.025 --act-lpf 0.7 --video`: the continuous gait at p4's
   shift, on time.
2. `python policy_stand.py run --run x3_g13Ls --enable --vx 0.025 --act-lpf 0.7 --video`: the same with shove
   training. Does a small balance loss settle instead of escalating?
3. `python policy_stand.py run --run x2_g10L --enable --vx 0.025 --act-lpf 0.7 --video`: less shift (0.10). Robust in
   simulation but in the late-shift zone. Watch the first steps; it tests the timing check.
4. Baseline, same session: `python policy_stand.py run --run p4_gate --enable --vx 0.03 --act-lpf 0.7 --video`.

What to watch:
- Does the body move forward continuously instead of surging in each step?
- Is the step size closer to k5's?
- After a wobble, does the next step settle or escalate?
- Falls per step against p4.

## On the robot (2026-11-18 00:07-00:24, 26 trials)

Logs `stand_*_20261118_*.csv` (not committed). Analysis: `scripts/robot19_analysis/gait_metrics.py`; per-trial roll
per step in the scratchpad.

| policy | trials | walked / falls | stride | s / step | speed | body roll p2p per cycle | legs apart at lift-off |
|---|---|---|---|---|---|---|---|
| x1_g13L `--vx 0.025` | 16 | 143 s, 116 touchdowns; 12 falls (10 over the left foot), median 9 steps per trial | 45 mm | 1.22 | 19 mm/s | 9.1 deg | 13.4 deg |
| x2_g10L `--vx 0.025` | 3 | 34 s, 27 td; 1 fall | 40 | 1.22 | 17 | 13.9 | 12.6 |
| x3_g13Ls `--vx 0.025` | 4 | 15 s; 4 falls, all on step 3 (3 forward) | 26 | 1.31 | 10 | 8.4 (pitch 11.0) | 12.1 |
| p4_gate `--vx 0.03` | 3 | 73 s, 56 td; 0 falls (stopped) | 40 | 1.26 | 16 | 12.9 | 17.7 |
| l9_s18 `--vx 0.015` | 2 | 34 s; 0 falls | 49 | 1.24 | 20 | 15.1 | 14.5 |
| *k5_high2, 11-15/16 (benchmark)* | *57* | *844 s, 676 td* | *53* | *1.20* | *22* | *12.4* | *7.7* |

The operator:
1. x2 is the most reliable but adds nothing new; it looks like l9. x3 reacts to small flexes and twitches in normal
   walking (shove training).
2. x1 gets the timing most accurately, with butter-smooth steps, but falls after ~5 steps: the shifting builds up
   lateral instability.
3. p4's hip sway is too much.
4. Smooth steps need step timing, not big hip sways. p4 and the x policies give up too much speed, and the sway
   itself costs stability.
5. Next: back to k5 (its speed and step size are a hard requirement), l9 the benchmark; make k5's steps continuous and
   stable through the reference gait.

**How x1 falls.** The sideways rock grows by about 1 deg per cycle until the robot goes over the left foot, on a
right swing, 9-10 s in (11-18 00:24:39, roll in deg, + = toward the right):

| lift-off | R 0.4 s | L 1.6 | R 2.8 | L 4.0 | R 5.2 | L 6.4 | R 7.6 | L 8.8 | R 10.0 |
|---|---|---|---|---|---|---|---|---|---|
| roll at lift-off | +1.2 | -2.3 | +2.5 | 0.0 | +4.3 | -1.5 | +5.3 | -2.1 | +3.4, then -20 |
| roll range in the step | -2..+3 | -4..+2 | -1..+3 | -5..+4 | -1..+7 | -6..+5 | -2..+9 | -6..+3 | fell |
| legs apart, robot / table | 9 / 15 | 13 / 15 | 17 / 15 | 12 / 15 | 19 / 15 | 13 / 15 | 19 / 15 | 14 / 15 | 18 / 15 |

- **The shift is lopsided.** On right swings the legs open 17-19 deg (table 15), on left swings 12-13. The policy adds
  +9 deg on the right hip roll at each right lift-off, +6 on the left.
- **Each right step the body droops further toward the swing side** (+1 -> +5 deg at lift-off, +3 -> +9 peak). The
  correction swings it back over the left foot, and once it overshoots it falls outward over the stance foot.
- The same trace in 3 of 3 falls checked (00:07:29, 00:12:05, 00:24:39). Every policy walks with a 1-2 deg right bias
  on this robot; with a big shift, the bias compounds instead of averaging out.

**x3:** shove training made it react to every small flex. 4 of 4 trials fell on the third step, 3 forward (pitch
range 11 deg per cycle against x1's 6).

**What the numbers say about speed.** Every policy steps every 1.2-1.3 s, so speed is stride: k5 53 mm, l9 49, x1
45, p4 40. k5's swing is 0.45 s from lift-off to touchdown; p4 / x tables 0.55 s, l9 0.6 s. k5 lifts off with the legs
only 7.7 deg apart: its policy cancels half of its table's 13.7 deg shift.
