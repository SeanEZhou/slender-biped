# Round 22: k5's steps, made continuous and steadier through the reference gait (2026-11-18 00:55-08:47, simulation)

Scripts: `scripts/robot22_common.sh`, `robot22_queue.sh` .. `robot22_queue6.sh`. Evaluations: `eval_<run>.txt`.

## Why

Robot round 21 (results/robot21/NOTES.md): x1 stepped on time and smoothly but its lopsided weight shift built a
sideways rock and it fell in ~9 steps; x2 walked like l9; x3 twitched. The operator: p4's hip sway is too much and
costs speed; smooth steps need step timing, not big sways. Go back to k5:
- k5's speed and step size are a hard requirement (robot: 53 mm strides, 22 mm/s; l9 49 mm, p4 40 mm);
- l9 is the benchmark;
- change the reference gait, not only the rewards:
  1. a continuous gait on k5's steps;
  2. landing timing;
  3. hip-sway timing, with less sway than p4;
- the swing (lift-off to touchdown) must not be longer than k5's 0.45 s.

## The tables

All from `shift_gait.py step` with k5's c_T12HH parameters:
- 1.2 s steps, a 0.45 s swing, lift 0.75 / 0.75 rad, lift-off 0.5 s into the step;
- so k5's stride at the same command, and the settle gate unchanged.

`--shift-lead` (new) finishes the shift that many seconds before lift-off.

| table | change from k5's | legs apart (table) at lift-off / touchdown |
|---|---|---|
| c_T12HH (k5) | the weight moves back toward the swing foot from lift-off on | 13.7 / 3.7 deg |
| c_K1g | continuous pelvis (`--glide`) | 13.7 / 3.7 |
| **c_K2gr** | glide, shift held until touchdown (`--rel0 1.0`) | 13.8 / 13.7 |
| c_K3gs10 | glide, held, shift 0.10 complete 0.1 s before lift-off | 11.5 / 11.4 |
| c_K4gs08 | as K3, shift 0.08 | 9.2 / 9.2 |
| c_K5gs10p | K3 + the foot over its spot by 80 % of the swing, then straight down | 11.5 / 11.4 |
| **c_Z1soft** | K2 + the foot rises over 35 % of the swing and lowers over 65 % (was 45 / 55) | 13.8 / 13.7 |
| c_Z2lead | K2 with the shift complete 0.1 s before lift-off | 13.8 / 13.7 |
| c_Z3t11 | K2 with 1.1 s steps (same 0.45 s swing) | 13.8 / 13.7 |
| c_Z4r08 | K2 with the release starting at 80 % of the swing | |
| c_K6r115 | K2 with the release starting 0.07 s after touchdown | |

For comparison: p4's c_P1s13 is 14.9 / 14.9 deg with a 0.55 s swing; l9's c_S9mid18 is 20.6 / 20.6 deg with a
0.6 s swing.

## Results (simulation)

Recipe:
- round 17's (b17), plus `land_h 0.02 land_v -100` and `late_shift -2000`;
- y runs from k5_high2 (16M) at 25M; z runs from the y run named, +20M.

Columns:
- up / cm in 20 s at 0.015 · 0.025 · 0.035 m/s, robot sim;
- loose and slow hips at 0.025;
- through shoves 0-0.1 m/s;
- legs apart at lift-off (p50, deg);
- landing speed p50 (mm/s) / jolt after touchdown.

| run | table, start | up / cm | loose | slow | shove | apart | landing |
|---|---|---|---|---|---|---|---|
| *k5_high2* | *c_T12HH* | *79 % 29 · 85 % 32 · 85 % 38* | *0 %* | *33 %* | *87 %* | *15.6* | *252 / 14.2* |
| *l9_s18* | *c_S9mid18* | *79 % 40 · 79 % 44 · 69 % 46* | *31 %* | *54 %* | *78 %* | *31.1* | *194 / 12.6* |
| y1_k1 | K1g, k5 | 83 % 27 · 92 % 33 · 81 % 40 | 2 % | 35 % | 88 % | 16.0 | 261 / 16.3 |
| **y2_k2** | K2gr, k5 | 90 % 37 · 88 % 43 · 88 % 46 | 17 % | 40 % | 88 % | 14.8 | 288 / 17.1 |
| y2b_k2s1 | K2gr, k5, seed 1 | 88 % 38 · 67 % 42 · 71 % 45 | 0 % | 29 % | 75 % | 14.8 | 250 / 14.2 |
| y3_k3 | K3gs10, k5 | 65 % 34 · 83 % 39 · 90 % 44 | 0 % | 33 % | 84 % | 14.6 | 272 / 14.3 |
| y4_k4 | K4gs08, k5 | 60 % 34 · 73 % 36 · 77 % 37 | 27 % | 27 % | 72 % | 9.7 | 268 / 15.9 |
| y5_k5 | K5gs10p, k5 | 31 % 41 · 52 % 49 · 52 % 51 | 0 % | 27 % | 69 % | 14.1 | 271 / 15.5 |
| y6_k6 | K6r115, k5 | 65 % 34 · 77 % 44 · 69 % 53 | 2 % | 17 % | 75 % | 15.4 | 271 / 14.2 |
| **y7_z1** | Z1soft, k5 | 88 % 37 · 88 % 41 · 83 % 44 | 33 % | 52 % | 84 % | 15.0 | 255 / 16.4 |
| z1_soft | Z1soft, y2 | 71 % 34 · 77 % 40 · 75 % 46 | 2 % | 35 % | 78 % | 17.6 | 249 / 14.3 |
| z2_lead | Z2lead, y2 | 79 % 32 · 88 % 38 · 79 % 43 | 15 % | 46 % | 86 % | 17.7 | 273 / 17.3 |
| z3_t11 | Z3t11, y2 | 40 % 35 · 33 % 45 · 44 % 50 | 0 % | 38 % | 55 % | 17.9 | 254 / 15.0 |
| z4_r08 | Z4r08, y2 | 23 % 38 · 25 % 47 · 40 % 52 | 2 % | 31 % | 39 % | 18.6 | 248 / 15.4 |
| **z0_y2c** | K2gr, y2 (control) | 96 % 32 · 96 % 38 · 94 % 41 | 29 % | 69 % | 95 % | 16.2 | 276 / 17.2 |
| **z0b_y2bc** | K2gr, y2b | 92 % 35 · 90 % 39 · 92 % 46 | 10 % | 42 % | 88 % | 18.0 | 259 / 13.9 |
| z7_y7c | Z1soft, y7 | 90 % 33 · 85 % 32 · 88 % 34 | 44 % | 56 % | 94 % | 16.0 | 260 / 17.8 |

z7 is the most robust to loose hips (44 %), but its distance no longer grows with the command (33 / 32 / 34 cm): it
has flattened the stride the way p4 did. Against the hard requirement, y7 is the slower-descent candidate, not z7.

## What it says

- **The gain is the shift held until touchdown (c_K2gr).**
  - k5's table moves the weight back toward the swing foot from lift-off, so the body falls onto the landing foot.
  - Held until the foot is down, on two seeds:
    - 20-35 % more distance than k5 at the same stride and swing (y2 37-46 cm, y2b 38-45, k5 29-38);
    - at least k5's robustness.
  - The legs open no more than k5's (15 deg at lift-off; l9 31).
- **A continuous pelvis alone (y1) changes little.** It is the hold that matters.
- **Less shift than k5's makes it less robust** (y3 0.10, y4 0.08). This is the opposite of p4's case, where 0.13
  was the most and the question was whether it was too much.
- **The release must start at touchdown:**
  - starting before it (z4, 80 % of the swing) falls (23-40 % up);
  - 0.07 s after it (y6) walks far at speed but less robustly.
- **A slower descent within the same 0.45 s swing (Z1soft) works when trained from k5:** y7 has the best loose-hip
  result of the round (33 %) and k5's landing speed. As a fine-tune from y2 (z1) it looked bad.
  - So tables have to be compared from the same start: a policy trained on one table and moved to another loses
    more than the table difference.
- **20M more on the same table makes the K2 policies robust on both seeds:** z0 94-96 % up, 69 % slow hips, 95 %
  through shoves; z0b 90-92 %, 88 %. They give back some distance (z0 32-41 cm) but stay at or above k5's.
- **Faster cadence (1.1 s steps, z3) failed;** so did the foot placed early and set straight down (y5).

## To test on the robot

All `--enable --act-lpf 0.7 --video`, today's leg. The tables ask k5's stride at the same command; k5 ran at 0.015
and overstepped to 53 mm. Run each at `--vx 0.02` and compare stride and speed against k5 (gait_metrics).

1. `python policy_stand.py run --run z0_y2c --enable --vx 0.02 --act-lpf 0.7 --video`: the steadiest in
   simulation; k5's step, held shift, continuous pelvis.
2. `python policy_stand.py run --run z0b_y2bc --enable --vx 0.02 --act-lpf 0.7 --video`: the same table, other seed;
   softer landings, farther.
3. `python policy_stand.py run --run y7_z1 --enable --vx 0.02 --act-lpf 0.7 --video`: the slower descent.
4. Baseline, same session: `python policy_stand.py run --run k5_high2 --enable --vx 0.015 --act-lpf 0.7 --video`.
5. Benchmark, same session: `python policy_stand.py run --run l9_s18 --enable --vx 0.015 --act-lpf 0.7 --video`.

Comparison (repo root), today's logs from 10:00 on:
`.venv/bin/python scripts/robot19_analysis/gait_metrics.py $(for f in hardware/leg_p0/software/stand_{z0_y2c,z0b_y2bc,y7_z1,l9_s18,k5_high2}_20261118_1*.csv; do case $f in *_frames.csv) ;; *) [ -f "$f" ] && [ $(wc -l <"$f") -gt 200 ] && echo "$f";; esac; done)`

What to watch:
- **Stride and speed:** at least k5's. If the steps are short, try `--vx 0.025`.
- Whether the body falls onto the landing foot (k5) or sets down over it.
- Sideways rock: does it stay small, or grow like x1's?
- Falls per step against k5.

## On the robot (2026-11-18 10:10-11:39)

Logs `stand_*_20261118_1*.csv` (not committed). z0, z0b, y7 at `--vx 0.02`; k5, l9 at 0.015; all `--act-lpf 0.7`.
"Stopped" = the operator ended the trial; a fall = the body past 20 deg.

| run | trials | steps | falls | longest walks | stride | speed | body roll per cycle | landing knee deg/s | jolt peak |
|---|---|---|---|---|---|---|---|---|---|
| **z0_y2c** | 8 | 140 | 2 (R at step 5, L at step 20) | 40 / 28 / 27 / 24 / 24 / 22 s | 52 mm | 22 mm/s | **11.0 deg** | 220 | 47 |
| k5_high2 | 5 | 52 | 3 (L, L, R at steps 5-11) | 19 / 17 s | 53 mm | 22 mm/s | 12.1 deg | 228 | 52 |
| l9_s18 | 5 | 54 | 1 (L at step 3) | 23 / 19 / 19 s | 40 mm | 16 mm/s | 15.6 deg | 57 | 33 |
| z0b_y2bc | 4 | 8 | 3 (steps 1-3) | 4 s | | | | 213 | 36 |
| y7_z1 | 2 | 4 | 2 (L at step 2) | 3 s | | | | 171 | |

The operator: z0 is an improvement over k5 (similar steps, but the body stays upright with less lean and shake), so
this direction is workable. l9's large hip movement looks unnatural even though it works; it is dropped as a direction.

- **z0 keeps k5's step** (stride and speed at 0.02 equal k5's at 0.015) **and rocks less:** ~7 deg per step against
  k5's 8-9 and l9's 10-11, steady over 20-30 steps. k5's falls came after its rock grew (11 -> 21 deg, 10 -> 25 deg);
  z0's didn't. Falls per step: z0 1.4 %, l9 1.9 %, k5 5.8 %.
- **z0 lands as hard as k5 because of servo lag.** The landing knee, mean over touchdowns (deg, + = bent):

  | ms from scheduled touchdown | -240 | -160 | -80 | 0 | +80 |
  |---|---|---|---|---|---|
  | z0 commanded | 64 | 54 | 41 | 32 | 32 |
  | z0 measured | 53 | 65 | 60 | 45 | 34 |
  | k5 measured - commanded | -10 | +10 | +20 | +14 | +4 |
  | l9 measured - commanded | +12 | +4 | +4 | +1 | +1 |

  - The knee and hip pitch run ~100 ms behind the command. At the scheduled touchdown the knee is still 12-14 deg
    more bent than commanded (hip pitch 8-11 deg), so the foot lands ~0.1 s late, while the servos are still catching
    up at ~210 deg/s.
  - The table's slow final approach (knee -70 deg/s over the last 60 ms) is shorter than the lag, so it never reaches
    the foot. The servo plays the table's mid-descent (~180 deg/s) at the moment the foot lands.
  - l9 is soft because its 0.6 s table is nearly flat for the last 240 ms (45 -> 34 deg). The servos keep up
    (<= 4 deg behind), and the foot arrives where and when the table says.
  - The policy doesn't correct this: its knee residual at touchdown is about +-1 deg on every policy.
  - y7's slower descent (Z1soft, -49 deg/s over the last 60 ms) is still inside the lag window. It fell in 2 steps
    both times.
- **What a soft landing at k5's 0.45 s swing needs:** the commanded knee nearly still for the last ~150 ms before
  touchdown (longer than the lag), or the descent commanded ~0.1 s early. Neither lengthens the swing. Check first
  whether the simulator's servo model has this ~100 ms lag (the training model has to see it).

- **The lean to the left: the policy, plus a fixed ~1.5 deg between the torso and the legs; no sign of loosening.**
  Standing before engaging, the body roll each foot's joint geometry asks for (that foot flat) against the IMU:

  | | table at u = 0 | commanded (table + action) | IMU |
  |---|---|---|---|
  | z0 (7 trials) | 0 / 0 | L foot -3.1, R foot +3.9 | -2.0 to -3.5 |
  | k5 today (5) | 0 / 0 | -0.2 / -1.8 | -2.2 to -4.3 |
  | k5 11-15/16 (5) | 0 / 0 | -0.5 / -0.2 | -1.4 to -2.3 |
  | z0b (3) | 0 / 0 | +0.3 / +0.6 | -2.9 to -3.3 |

  - z0's start lean is commanded. Its standing action rolls the feet 7 deg apart, the robot stands flat on the left
    foot, and the IMU reads what the left leg's geometry asks for. The table itself is symmetric.
  - k5 and z0b command a level stance, yet the IMU reads 1.5-3 deg further left than either foot allows. This excess
    was 1-2 deg on 11-15/16 as well, so it is a standing offset that hasn't grown.
  - Walking: on one foot, the body sags toward the swing side by ~9 deg more than the joints say (unseen compliance).
    Measured, it is 7.7 deg on left stance and 10.5 deg on right stance (k5). Subtracting a fixed 1.5 deg offset
    makes both sides equal, and the left-stance value has been the same since 11-15 (7.7 -> 7.9). So nothing on the
    left has loosened over these four days.
  - The policies' hip-roll residuals are symmetric (+ = foot outward on both legs; z0 +5.3 left, +4.9 right).
  - Check the torso-to-pelvis mount and the IMU board. A torso sitting ~1.5 deg left of the pelvis would give
    exactly this.

Side-by-side video, first 4 steps, the robot's full body (scratchpad only, never committed):
`walk22_z0_k5_l9.{gif,mp4}` (real time) and `walk22_halfspeed_z0_k5_l9.{gif,mp4}`.
