# Round 17: steps that set down instead of crashing (2026-11-16 02:00-08:52, 6.8 h of the 8 h GPU)

The operator after testing round 16 (m2, m2b, m1, k5): the new policies look like k5. "The robot still feels like it
risks falling on every step and doesn't evoke a sense of confidence ... every step comes crashing down very quickly
and vibrates the whole body." Goal: clear the rug, but land and walk gracefully, not stomp.

## The round-16 robot trials (2026-11-16 01:43-01:49, 8 trials)

[walk_steps.py](../../scripts/robot16_analysis/walk_steps.py): k5 2 trials (both fell R), m1 1 (fell R), m2 3 (2 stopped
up at 21 / 28 s, 1 fell L), m2b 2 (1 fell R at 7 s, 1 stopped up at 29 s). Too few to rank; m2 looked best.

## Every touchdown is a jolt (robot and simulator)

[landing.py](../../scripts/robot17_analysis/landing.py): the gyro's tick-to-tick change (roll and pitch rate, deg/s
per 20 ms), from the robot's log. k5's 00:39 batch (the best one), by cycle fraction:

| | after a lift-off (0.21-0.37) | mid double support | **0.1 s after a touchdown** (0.42-0.50, 0.92-1.0) |
|---|---|---|---|
| jolt RMS | 7-9 | 7-10 | **21-28** |

The same for all four policies tonight (RMS over the 0.2 s after each touchdown: k5 25, m1 27, m2 23, m2b 24).

The GPU simulator ([gait_eval.py](../../scripts/gait_eval.py), robot sim, 20 s at 0.015 m/s) shows it too: k5's boards
land at **260 mm/s** (p90 320), and the jolt after a touchdown is 2.8x mid-swing (14.6 vs 5.2).

## Why: c_T12HH's design

c_T12HH ([shift_gait.py](../../slender_biped/shift_gait.py), reproduced exactly from its params): per 1.2 s step,
settle 0.05 s, weight shift 0.45 s, **swing 0.45 s**, then 0.25 s release and 0.5 s more in double support.

1. **The weight shift unwinds from lift-off** (over the swing + 0.25 s). By touchdown 72 % of it is gone: the reference
   moves the pelvis back toward the swinging foot while the robot stands on one leg. The body falls onto the new foot.
2. **The swing asks for more than the servos have.** 0.75 rad of knee and hip lift up and down in 0.45 s: the
   reference needs up to 87 rad/s^2 of joint acceleration; the servos' profile ramps at 35-60. They lag, and the foot
   arrives while still moving.
3. **Double support takes 0.75 s of every 1.2 s step** (62 %; people walk with ~20 %), so there is time to give to the
   swing without slowing the cadence.

## What others do (searched 2026-11-16)

- Humanoid walking-pattern generators plan the swing foot to reach zero velocity (and zero acceleration) at
  touchdown, e.g. Honda's landing-impact patents, SURENA III's polynomial swing, sixth-order swing polynomials on torque
  humanoids. A soft-landing strategy adds an online admittance correction (Industrial Robot 2017).
- RL walkers penalize landing directly: TactileStep (Unitree G1, arXiv 2609.28959) penalizes downward foot speed above
  0.24 m/s and downward acceleration before landing, then the peak normal force (by body weight) and its rate; real
  impact force down 5-49 % by terrain, success rate kept. VMTS (arXiv 2503.07049) penalizes foot vertical speed near
  the ground because simulated forces are unreliable. Disney's Olaf adds an impact-reduction reward because harsh
  footsteps break the character.
- People: the foot slows right before contact (heel speed changes sign at foot strike), and the toe's lowest point is
  mid-swing (minimum toe clearance, Winter 1991).

## What round 17 tries

New table options in `shift_gait.py` (defaults reproduce c_T12HH):
- `--set-frac F --hover H`: the swing drops to H of the lift, then sets down over the last F of the swing, slowly (zero
  speed at both ends).
- `--place-frac P`: the foot reaches its fore-aft spot by P of the swing, then comes straight down.
- `--rel0 R`: the weight shift starts to release at R of the swing (1 = at touchdown: the body stays over the stance
  foot until the new foot is down, then moves over in double support).

| table | swing | lift (knee, hip rad) | set-down | weight shift released |
|---|---|---|---|---|
| c_T12HH (k5) | 0.45 s | 0.75, 0.75 | none (half-cosine down in 0.25 s) | from lift-off |
| c_S1hold | 0.45 s | 0.75 | none | after touchdown (0.35 s) |
| c_S3long | **0.60 s** (lift-off at 0.40 s, u 0.1667) | 0.75 | to 20 % by 72 % of the swing, then 0.17 s down | from lift-off |
| c_S4lh | 0.60 s | 0.75 | same | after touchdown |
| c_S5mid | 0.60 s | **0.60** | same | after touchdown |

The longer swing halves the knee speed in the last 40 ms of the swing (1.0 vs 2.35 rad/s).

Reward: `land_v` (downward speed of a board centre below `land_h` = 2 cm, squared) at -100.

**Untrained check** (m2 on each table, [zero_shot.txt](zero_shot.txt)): landing speed 263 mm/s on c_T12HH, 180 on
c_S3long / c_S5mid, 289 on c_S1hold; jolt after touchdown 14.6 -> 13.1-13.8. The long swing works as designed, but the
policy needs training on it.

## Runs (each 20M from m2_attland 14M on its recipe, one lane at a time)

| run | table | land_v |
|---|---|---|
| `l1_landv` | c_T12HH | -100 |
| `l2_long` | c_S3long | -100 |
| `l3_lhold` | c_S4lh | -100 |
| `l4_mid` | c_S5mid | -100 |
| `l5_wide` | c_S6wide (0.6 s soft swing, held shift **0.18**) | -100 |
| `l6_wshort` | c_S7wshort (k5's 0.45 s swing, held shift 0.18) | -100 |
| `l7_s15` | c_S8mid15 (c_S5mid with shift 0.15), **from l4_mid** 20M | -100 |
| `l4b_mid` | c_S5mid, l4_mid + 20M | -100 |
| `l8_nolv` | c_S5mid, from m2, no land_v | 0 |
| `l9_s18` | c_S9mid18 (c_S5mid with shift 0.18), from l7_s15 | -100 |
| `l10_jolt` | c_S8mid15, from l7_s15, `ang_acc` x2 (-0.012) | -100 |
| `l11_hover` | c_S10hov (c_S8mid15, hover 30 %, set down over the last 42 %, faster rise), from l7 | -100 |
| `l12_rel` | c_S11rel (c_S8mid15, the weight moves over in 0.6 s after touchdown, was 0.35), from l7 | -100 |
| `l13_lv2` | c_S8mid15, from l7_s15 | **-200** |

Queues: `scripts/robot17_queue.sh` .. `robot17_queue6.sh`, settings `scripts/robot17_common.sh`. One lane at a time
(3.4-4.4 GB each), `mem_guard.sh` on throughout; it never fired.

### Why the jolt stays: the body is rolling into the landing (CPU sim, [td_cpu.py](../../scripts/robot17_analysis/td_cpu.py))

At contact the body still leans 4.5-6 deg over the stance foot but is rolling **toward the landing foot at ~26 deg/s**
(m2 26, l2_long 29, l3_lhold 26); the contact stops it (roll-rate change 22-28 deg/s). On the robot, too, the jolt
after a touchdown is mostly roll (24 deg/s per tick, pitch 12-14) plus a yaw twist (28-37).

The reference's centre of mass during a swing (MuJoCo, standing pose): with shift 0.12 it is only **1.0 cm inside the
stance board's inner edge** (1.6 cm from its centre); c_T12HH's unwinding shift puts it 0.6 cm *outside* by late swing.
With the hip roll's give under one-leg load, the body tips toward the swing side. Shift 0.18 puts the centre of mass
over the board's centre (hip roll 10.3 deg of its 14).

## Results (simulation)

[gait_eval.py](../../scripts/gait_eval.py), 20 s at 0.015 m/s, 48 runs (m2: 32); walk test = push_eval walk (14 s,
shoves 0 / 0.1 m/s, 64 runs). Landing speed = board centre's downward speed over the 20 ms before contact; jolt =
gyro roll/pitch change per 20 ms tick, RMS over the 0.1 s after a touchdown. Files: `eval_<run>.txt`.

| run | robot sim up (20 s / walk test) | loose sim up (20 s / walk test) | landing speed p50 / p90 mm/s | jolt after td (mid-swing) | clearance p10 / p50 mm |
|---|---|---|---|---|---|
| m2_attland (round 16) | 97 % / 93 % | - / 13 % | 263 / 323 | 14.6 (5.7) | 9.9 / 36.1 |
| `l1_landv` | 75 / 70 | 2 / 2 | 246 / 304 | 13.3 (5.1) | 26.6 / 39.8 |
| `l2_long` | 77 / 77 | 0 / 2 | **160** / 243 | 13.5 (4.4) | 31.5 / 46.2 |
| `l3_lhold` | 83 / 77 | 10 / 20 | 196 / 279 | 14.3 (4.5) | 3.0 / 32.9 |
| **`l4_mid`** | **100 / 97** | **27 / 34** | 184 / 253 | **12.2** (3.8) | 3.6 / 27.8 |
| `l5_wide` | 4 / 16 | 0 / 0 | 151 / 264 | **10.8** (4.3) | 2.9 / 41.4 |
| `l6_wshort` | 48 / 56 | 6 / 11 | 272 / 326 | 14.5 (5.5) | 8.1 / 35.0 |
| **`l7_s15`** (from l4_mid) | 90 / 91 | **40 / 52** | 177 / 257 | 12.0 (3.5) | 11.7 / 34.2 |
| `l4b_mid` (l4_mid + 20M) | 92 / 92 | 10 / 22 | 171 / 247 | 11.6 (3.7) | 14.0 / 30.8 |
| `l8_nolv` (c_S5mid, no land_v) | 98 / 98 | 0 / 5 | 218 / 272 | 13.6 (3.7) | 14.5 / 30.0 |
| `l9_s18` (from l7) | 79 / 80 | 44 / 45 | 192 / 270 | 12.4 (3.1) | **25.2** / 35.7 |
| `l10_jolt` | 90 / 86 | 38 / 43 | 176 / 257 | 13.6 (3.6) | 25.1 / 34.2 |
| `l11_hover` | 79 / 79 | 17 / 27 | 182 / 269 | 15.3 (4.7) | 12.3 / 33.9 |
| `l12_rel` | 50 / 62 | 4 / 12 | 194 / 266 | 14.7 (3.8) | 23.5 / 34.9 |
| `l13_lv2` | 69 / 69 | 0 / 10 | 192 / 275 | 13.7 (4.0) | 21.3 / 32.9 |

- The landing penalty alone (l1) barely softens k5's landing and costs stability.
- The 0.6 s soft swing cuts landing speed by a third (l2-l5); the jolt follows only with the held shift and the lower
  lift (l4).
- **l4_mid is the best: steadier than m2 in both sims and a softer landing.** Its full swings (CPU sim,
  [swings_cpu.py](../../scripts/robot17_analysis/swings_cpu.py)) clear p10 10.9 / p50 30.3 mm (m2 23.7 / 36.1); 7 % under
  10 mm (m2 1 %).
- **l7_s15 (shift 0.15, grown from l4_mid) is the most robust: the loose sim (round 11's model of k5's robot falls)
  40 % / 52 % up, every earlier policy <= 34 %.** It also walks further (39.5 cm in 20 s) and clears higher.
- 20M more on the same table (l4b) adds nothing; the landing penalty matters on the soft-swing table (l8 without it:
  landing 218 mm/s, loose sim 0 / 5 %).
- The wide shift (0.18) gives the softest landings but falls sideways (toward the swinging foot) after 20M from m2;
  with k5's short swing (l6) it doesn't help at all.

- Shift 0.18 grown from l7 (l9) walks, unlike l5: the best clearance and the softest jolt in the loose sim, less steady
  on the robot sim.
- Didn't help: the doubled jolt penalty (l10), a higher hover with a faster rise (l11: more mid-swing shake), a slower
  weight transfer after touchdown (l12: the body hangs off balance), a doubled landing penalty (l13).
- Every "more training from a good policy" lane regressed (l4b, l10, l13; m2b in round 16).

### Body motion at touchdown (CPU sim, [td_cpu.py](../../scripts/robot17_analysis/td_cpu.py), 6 x 20 s)

| | roll rate toward the landing foot before contact | roll-rate change at contact | body roll at contact (over the stance foot) |
|---|---|---|---|
| m2_attland | 26 deg/s | 28 | 4.5 deg |
| l3_lhold | 26 | 26 | 6.0 |
| l4_mid | 26 | 19 | 6.3 |
| l7_s15 | 23 | 18 | 7.2 |
| l9_s18 | 25 | 19 | 7.1 |

The held, wider shift keeps the body further over the stance foot and the stop at contact is a third smaller, but the
body still rolls toward the new foot at ~24 deg/s before it lands: the next thing to fix.

## Final ranking ([final_rank.txt](final_rank.txt): 96 runs per row, a fresh seed)

| | robot sim up | loose sim up | walked in 20 s | landing speed p50 / p90 | jolt after td (mid-swing) | clearance p10 / p50 | at 0.03 m/s: up, walked |
|---|---|---|---|---|---|---|---|
| k5 | 72 % | 0 % | 27 cm | 259 / 313 | 15.3 (5.4) | 23.9 / 41.6 | |
| m2_attland | 91 % | 3 % | 29 cm | 260 / 320 | 14.7 (5.7) | 15.9 / 36.4 | 88 %, 39 cm |
| **l4_mid** | **98 %** | 28 % | 34 cm | 188 / 250 | 12.3 (3.8) | **5.0** / 28.2 | **93 %, 50 cm** |
| **l7_s15** | 91 % | **36 %** | **39 cm** | **180** / 256 | 12.2 (3.5) | 11.3 / 33.6 | 88 %, 50 cm |
| l9_s18 | 81 % | 36 % | 39 cm | 184 / 267 | **11.8 (3.0)** | **24.5** / 36.1 | 74 %, 44 cm |
| l13_lv2 | 71 % | 3 % | 36 cm | 187 / 268 | 13.1 (3.9) | 22.2 / 34.2 | 51 %, 45 cm |

Against m2 (round 16's best), l7_s15 and l4_mid:
- land 30 % slower (180-188 against 260 mm/s);
- jolt 17 % less after a touchdown and 35-40 % less mid-swing (3.5-3.8 against 5.7);
- stay up as well or better on the robot sim, and 10x as often on the loose sim (28-36 % against 3 %);
- walk 20-35 % further at the same command, and 25 % further than m2 at 0.03 m/s.

The catch: **l4_mid's lowest swings are low** (p10 5 mm, 7 % of full swings under 10 mm on the CPU sim): watch for
toe catches on the rug. l7 (p10 11 mm) and l9 (p10 25 mm) lift more.

## To test on the robot (best first)

The new gaits are in each run's config (table, gate timing, attitude gate); the runner needs nothing new.

1. `python policy_stand.py run --run l7_s15 --enable --vx 0.015 --act-lpf 0.7 --video` (the most robust, soft landings)
2. `python policy_stand.py run --run l4_mid --enable --vx 0.015 --act-lpf 0.7 --video` (the steadiest on the robot
   sim; lower swings)
3. `python policy_stand.py run --run l9_s18 --enable --vx 0.015 --act-lpf 0.7 --video` (the highest, gentlest steps;
   less steady in simulation)
4. Baseline in the same session: `python policy_stand.py run --run m2_attland --enable --vx 0.015 --act-lpf 0.7 --video`
5. If 1 or 2 walk well, the same with `--vx 0.03` (twice the speed; in simulation they keep their landings).

What the steps should look like: a longer, slower swing (0.6 s of the 1.2 s step, against 0.45), the foot slowing
and setting down rather than dropping, and the body staying over the standing foot until the new one is down, then
moving across. The sideways sway is larger (the shift is 0.15 rad against 0.12).

Afterwards: `scripts/robot17_analysis/landing.py <logs>` gives the robot's jolt after touchdown (k5 / m2 23-25 RMS
over 0.2 s); `scripts/robot16_analysis/walk_steps.py` assumes c_T12HH's timing (lift-off at 0.208, touchdown at
0.396), so its per-step phases are off by 0.04 for these runs.

## On the robot (2026-11-16 10:18-10:25, 9 trials, `--vx 0.015 --act-lpf 0.7`, video)

The operator:
- **l9_s18** "best captures the goal of silent, soft, efficient steps", but "frequently steps just slightly above
  ground, so each step feels like the robot is tripping a little".
- **l7_s15** steps softer than k5 but is less stable than k5, and not as soft as l9.
- **l4_mid** trips after 1-2 steps.
- **m2_attland** is like k5.

| | trials | how they ended | touchdowns | jolt after touchdown, RMS p50 / p90 | jolt peak p50 | gyro peak p50 |
|---|---|---|---|---|---|---|
| k5_high2 | 1 | fell left at 11 s | 9 | 20.8 / 26.4 | 43.8 | 50 deg/s |
| m2_attland | 1 | stopped up at 28 s | 23 | 25.2 / 28.1 | 44.0 | 59 |
| l4_mid | 2 | fell forward at 11 s; stopped at 6 s | 12 | 11.6 / 20.9 | 20.7 | 36 |
| l7_s15 | 3 | fell back at 21 s, right at 10 and 14 s | 35 | 21.0 / 26.5 | 35.8 | 42 |
| **l9_s18** | 2 | **fell back at 41 s; stopped up at 48 s** | **70** | **20.3 / 27.8** | **31.0** | **43** |

([landing.py](../../scripts/robot17_analysis/landing.py); jolt = the gyro's tick-to-tick change, deg/s per 20 ms,
over the 0.2 s after a touchdown.) **l9: 1 fall in 70 touchdowns; its peak jolt is 30 % below m2 / k5.** Too few
trials for firm falls-per-step numbers. l4's low jolt is over only 10 steps before it tripped.

### Why l9 "trips": the swing ends skimming the rug

Video (102357, both feet): through the last third of every swing, the board hangs flat a few millimetres above the rug
while it is still moving. It brushes the pile before it is in place, then sets down.

That is the table. c_S9mid18's lowest board point over its swing, by the gait clock: 0 2 9 24 38 44 36 20 9 6 4 2 0 mm.
- **The set-down ends ~0.25 s within 10 mm of the floor.** That's the "soft" part: the hover at 0.2 of the lift and the
  slow last 28 %.
- **The lift is an equal hip + knee bump.** It keeps the board level, but the foot moves on a circle around the hip.
  Lifting swings the foot ~5 cm forward; lowering pulls it back. The last 20 mm of descent is mostly a backward sweep
  just above the floor.

On the robot the foot also trails the command by ~0.1 s
([swing_clear.py](../../scripts/robot18_analysis/swing_clear.py), encoder FK, every run).
- **Robot:** the median swing peaks at 47 mm (as the table) but reaches its peak 0.05-0.1 of the swing late.
- **Simulator** (`gait_eval --profile`): l9 is down at 83 % of the swing, ~0.14 s before the clock's touchdown.
- **Why they differ:** the encoders can't see the hidden flex.

### The tables jump at every step boundary (a round-17 bug)

In c_S5mid / c_S8mid15 / c_S9mid18 (`rel0 1`, `release_t 0.35`), the release after touchdown is still 39 % unfinished
when the step ends (0.2 s after touchdown). The next step starts from zero, so the hip-roll targets jump 3.0 / 3.7 /
4.4 deg in one 9 ms table step, both legs, twice a cycle (max per-phase change 0.052-0.077 rad; elsewhere 0.007). The
policies (residual 0.25, `--act-lpf 0.7`) smooth part of it. Still, it's a built-in jolt 0.2 s after every touchdown,
largest on l9.

### Landings

[walk_steps.py](../../scripts/robot16_analysis/walk_steps.py), now with each run's own table timing.

On l9:
- The body leans 3.5-5 deg toward the landing foot at touchdown, still rolling into it at ~30 deg/s (p90 47-64).
- Both boards land on their inner edge: 3 deg median, 44 % / 17 % (L / R) beyond 5 deg.
- That's flatter than k5 (4 deg, half beyond 5 deg).

### A correction to rounds 16-17: the right foot steps first

The reference tables' stance columns are [left, right], and the right foot swings first (lift-off u 0.208, touchdown
0.396 on c_T12HH; checked on the model and the robot's encoders). walk_steps.py and landing.py had the sides swapped
(fixed). Most round-17 numbers are per run and unaffected. Round 16's side-specific findings are corrected in
[robot16/NOTES.md](../robot16/NOTES.md).

## Verdict

**l9_s18 goes on the working-policy list**: the softest steps so far, 41 s and 48 s trials. Its weakness is in the
table (the skim and the jump), which is what round 18 rebuilds
([robot18/NOTES.md](../robot18/NOTES.md)).
`python policy_stand.py run --run l9_s18 --enable --vx 0.015 --act-lpf 0.7 --video`
