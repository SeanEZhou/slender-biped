# Round 19: less lean, and no weight transfer before the foot is down (2026-11-16 19:25-21:09, 2 h GPU)

## Round 18 on the robot (2026-11-16 18:52-19:01, 17 trials, `--vx 0.015 --act-lpf 0.7`, video)

| policy | trials | falls | touchdowns | per fall | in the simulator (robot / loose) |
|---|---|---|---|---|---|
| l9_s18 | 4 | 3 | 57 | 19 | 75 / 42 % |
| n15_s20 | 8 | 7 | 32 | 4.6 | 91 / 86 % |
| n19_s25 10M | 2 | 2 | 6 | 3 | 88 / 79 % |
| n18_s23 10M | 2 | 2 | 3 | 1.5 | 80 / 66 % |
| n7_s12 | 1 | 1 | 3 | 3 | 83 / 61 % |

The operator: n15 is quieter than l9, but every new policy falls within a few steps.
1. **The step feels "done" mid-air:** the robot is already leaning before the foot is down, and the lean kills it.
2. **n15 leans far more than it needs to.** It seems to focus on leaning rather than stepping.
3. **The leaning makes steps longer, and they are shorter than l9's.**

What the logs show ([r19 analysis in this file's scripts](../../scripts/robot18_analysis/)):

- **Falls start right after a touchdown** (0-0.34 s after the right foot's) **or late in a swing** (0.38-0.46 s after the
  left foot's lift-off), tipping **outward over the stance foot**.
  - n7 and n19 fall the same way.
  - The CPU simulator's round-18 falls were the same (outward, corrected above).
- **The lean:**
  - At lift-off, n15's legs open ~30 deg: the swing hip roll +19-21 deg out, the stance hip in 8-10 deg.
  - l9: ~20 deg (+15-19 out, only 3.5-4.4 in): l9's policy cancels most of its table's 10 deg stance shift.
  - n15 keeps all of it.
- **Stops:** on the robot the gate held n15's clock 14-31 % of the time (simulator 5 %) [wrong, see the correction below: 5.6 %], so its steps take 1.28-1.35 s
  against l9's 1.23-1.28. The lean trips the gate's lean check.
- **The landing:** n15's knee extends at ~193 deg/s just before touchdown (l9 ~70): the straight-down drop is fast on
  the robot. The peak jolt after touchdown is the same as l9's (33 vs 34). The operator hears it as quieter.
- **The robot's joints trail their commands** (best-lag fit, cmd vs encoder, every trial):
  - knee 80 ms;
  - hip pitch 80 ms;
  - hip roll 120-140 ms.
  - The release of the shift starts at the scheduled touchdown, so it starts with the foot still in the air. That is
    point 1.
  - The CPU simulator's ROBOT settings: knee 80, hip pitch 40, hip roll 60 ms. Training uses slower hip ramps (`prof_acc
    [7, 35, 35]` x U(0.6, 1.5)), so this is not a training gap. Hips twice as stiff, or a looser left hip, didn't
    reproduce the robot's falls in simulation.

## What round 19 tries

Tables (`shift_gait.py step`):
- the shift held **0.1 s past the scheduled touchdown**, then released over 0.2 s (`rel0 1.18`, `release_t 0.3`);
- no centre pause;
- a 0.35 s shift;
- a 0.55 s swing;
- the 1.2 s step kept;
- **shift 0.13** (from 0.18): lean only as much as the step needs.

| run | table | swing | shift | from |
|---|---|---|---|---|
| p1_s13 | c_P1s13 | n15's (retraction high, straight down) | 0.13 | n15 |
| p3_bump | c_P3bump | l9's bump | 0.13 | n7 (l9 + the jump fix) |
| p2_s18 | c_P2s18 | n15's | 0.18 (only the hold changes) | n15 |

## Results (simulation; gait_eval n48 at 0.015 m/s, 20 s; "slow hips" = prof_acc [6, 60, 25], the robot's measured lags)

Hip roll at lift-off from the CPU sim ([lo_cpu.py-style check]: commanded hip rolls, 3 x 12 s):

| run | robot sim up | loose sim up | slow-hip sim up | landing p50 | jolt after td (mid-swing) | clearance p10 / p50 | gate holding | hip roll at lift-off, swing / stance |
|---|---|---|---|---|---|---|---|---|
| l9_s18 | 75 % | 42 % | 67 % | 184 | 11.9 (3.0) | 22 / 36 | 7.7 % | 20 / 4 deg |
| n15_s20 | 91 % | 86 % | 88 % | 267 | 15.1 (2.4) | 35 / 43 | 5.4 % | 20 / 8 deg |
| p1_s13 | **96 %** | 85 % | 79 % | 277 | 19.6 (3.3) | 31 / 37 | 9.3 % | **17 / 6 deg** |
| p3_bump | 94 % | 69 % | 75 % | **173** | **13.4 (3.3)** | 24 / 32 | 11.2 % | 17 / 4 deg |
| p2_s18 | 65 % | 67 % | 71 % | 283 | 17.9 (3.0) | 35 / 40 | 5.6 % | 19 / 9 deg |
| p4_gate | 88 % | 73 % | 81 % | 269 | 19.8 (2.9) | 31 / 36 | **4.4 %** | 17 / 5 deg |

- **p1 (shift 0.13, transfer held 0.1 s past touchdown):** the legs open ~23 deg at lift-off, against n15's 28 and l9's
  24. It is as robust as n15 in simulation.
- **The cost:** the jolt after touchdown is up (19.6), and the gate holds more often in simulation (9.3 %).

- **p3 (l9's swing, shift 0.13, transfer held; from n7):** l9's soft landings (173 mm/s, jolt 13.4) and the least lean
  (legs open ~20.5 deg). Robot sim 94 %, but loose 69 %. Late in the swing it still comes close to the floor
  (9 / 2 / 0 mm). The gate holds 11 %.
- The gate holds 9-14 % of the time in the new runs, and on the robot it held n15 14-31 % [corrected: 5.6 %]. **p4_gate:** p1 with the
  gate's lean limits relaxed from 3 / 6 deg to 5 / 8 deg (`settle_gate_att [0.087, 0.140]`), from p1. Fewer stops,
  if the lean is now small enough not to need them.
- **p2 (only the transfer held, shift 0.18 kept): worse** (65 / 67 / 71 %, the legs still open 28 deg). The smaller
  shift is what helps p1 and p3, not the held transfer alone.
- **p4 (p1 with the gate's lean limits relaxed to 5 / 8 deg, from p1): the clock stops half as often** (4.4 %, against
  p1's 9.3 %). It is a little less robust in simulation (88 / 73 / 81 %).

## Verdict

- **Less lean is the change that helps.** Shift 0.13 (p1, p3) against 0.18 (p2, n15): the legs open 20-23 deg at
  lift-off instead of 28, and p1 / p3 stay up 94-96 % in the robot sim (p2: 65 %).
- **Holding the transfer 0.1 s past the scheduled touchdown alone doesn't help** (p2). It is in all four tables, so on
  the robot it should keep the body over the stance foot until the late-arriving foot is down (operator's point 1).
- **The simulator overrates new policies.** Round 18's n15: 91 % in simulation, a fall every ~5 steps on the robot.
  Treat these numbers as a screen, not a prediction. The robot decides.

## To test on the robot (best first)

All `--vx 0.015 --act-lpf 0.7 --video`. Each run's config carries its table, gate timing and gate limits.

1. `python policy_stand.py run --run p3_bump --enable --vx 0.015 --act-lpf 0.7 --video`
   - l9's swing (the one that walks on the robot), the least lean (legs ~20 deg apart at lift-off), the transfer held
     past touchdown;
   - soft landings (173 mm/s in simulation);
   - it still comes close to the rug at the end of the swing, as l9.
2. `python policy_stand.py run --run p1_s13 --enable --vx 0.015 --act-lpf 0.7 --video`
   - n15's quieter, rug-clearing swing with less lean (~23 deg) and the held transfer;
   - harder landings in simulation.
3. `python policy_stand.py run --run p4_gate --enable --vx 0.015 --act-lpf 0.7 --video`
   - p1 with fewer gate stops (relaxed lean limits): steps should come more evenly.
4. Baseline in the same session: `python policy_stand.py run --run l9_s18 --enable --vx 0.015 --act-lpf 0.7 --video`

What to look for:
- **Does the robot stay over the stance foot until the swinging foot is down?** Point 1: the transfer now waits 0.1 s.
- **Is the lean visibly smaller than n15's?**
- **Do steps come more regularly, with fewer pauses?** The log's `gate_held`; n15 was held 14-31 % on the robot [corrected: 5.6 %].
- **Falls per step against l9's 19 touchdowns per fall tonight.**

Afterwards: `scripts/robot18_analysis/swing_clear.py <logs>` (falls per touchdown, swing height) and
`scripts/robot17_analysis/landing.py <logs>` (jolt after touchdown).

## On the robot (2026-11-16 23:49 - 11-17 00:03, 19 trials)

User: p1 and p4 work reasonably well; there are still falls, but the steps look more coherent and stable. They are
slower than l9 or k5, whose steps are much bigger. p3 doesn't work.

`swing_clear.py` (fall = tilt > 20 deg) and `scripts/robot19_analysis/gait_metrics.py`. k5 and l9 rows are from earlier
sessions (11-15 21:37-21:51, 11-16 18:58-19:01) plus tonight's l9 baseline.


> **Correction (2026-11-17):** the robot log's `gate_held` counts the ticks a hold has lasted (0, 1, 2, ...); the
> round-18/19 robot numbers averaged that counter. The share of walking time the clock was held: k5 1.9 %, l9 4.3 %,
> n15 5.6 %, p1 2.5 %, p4 3.2 % (simulation figures were right: they test `gate_held > 0`). n15's lean did not trip
> the gate much more than l9's; its longer steps (1.30 s) are not explained by the gate.

| run | trials | walking s | touchdowns | falls | falls / td | step len mm | stride mm | s / step | speed mm/s | roll p2p deg | hip-roll spread at lift-off | gate held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k5_high2 | 12 | 196 | 157 | 11 | 7.0 % | 27 | 54 | 1.20 | 23 | 12.5 | 8 | 1.9 % |
| l9_s18 | 5 | 108 | 83 | 3 | 3.6 % | 25 | 42 | 1.24 | 17 | 13.3 | 17 | 4.3 % |
| n15_s20 (round 18) | 8 | 43 | 32 | 7 | 22 % | 21 | 24 | 1.30 | 9 | 8.6 | 27 | 5.6 % |
| **p1_s13** | 7 | 188 | 152 | 5 | 3.3 % | 26 | 34 | 1.22 | 14 | 9.4 | 16 | 2.5 % |
| **p4_gate** | 4 | 143 | 115 | 0 | 0 % | 27 | 34 | 1.24 | 14 | 10.3 | 17 | 3.2 % |
| p3_bump | 3 | 10 | 8 | 3 | 38 % | 17 | 30 | 1.22 | 12 | 8.6 | 12 | 3 % |

- **p1 / p4 are the steadiest walkers so far.** p4: 3 of 4 trials walked 45 s and were stopped up (one near-tilt
  of 10 deg each); p1: 2 x 50 s up, 5 falls in 152 touchdowns. Body roll per cycle is 9-10 deg against l9's 13 and k5's
  12.5, the gate holds 2.5-3.2 % of the time (n15 5.6 %, l9 4.3 %; corrected 2026-11-17), step time 1.22-1.24 s and even.
- **They are slower because they step exactly what they are asked.** At `--vx 0.015` every table asks for a
  36 mm stride (0.015 m/s x 2.4 s per cycle); p1 / p4 make 34 mm, at 14 mm/s. k5 (54 mm, 23 mm/s) and l9 (42 mm)
  overstep: l9 also in simulation (40 cm per 20 s against p1's 31), k5 only on the robot, where 5 of its 11 falls are
  forward - part of its stride is the body tipping forward and the step catching it.
- **Asking for more does not help yet.** In simulation p1 / p4 at vx 0.025 and 0.03 walk 33-37 cm per 20 s against 31
  at 0.015 (the table's stride doubles): the policy flattens the commanded stride
  (`results/robot20/eval_speed.txt`). Up stays 85-92 % (69-75 % with slow hips).
- **What is left of the falls is one direction: right.** 10 of the 11 tilts past 10 deg in p1 / p4 roll right, over
  the right foot: in p4 0.2 s after the right foot lands (the body keeps going right through the landing), in p1 during
  or after the left swing (the body drifts right over the right stance foot and the left landing cannot stop it).
  At the right touchdown the body is already 3-6 deg toward the right (p50, p4 5.7), at the left one 1-2.6 deg toward
  the left; every policy walks with a right bias (mean roll walking +1.3 to +2.2 deg against standing).
- **p3 trips.** Its swing peaks below 10 mm in the p10 swing and spends 38 % of mid-swing below 10 mm (10 % below 5):
  l9's bump swing with the shorter 0.55 s swing and place_frac 0.75 drags on the rug. 3 falls in 8 touchdowns.
- Landing: p1 / p4 knees still come down at 175-195 deg/s before touchdown (l9 66): quieter swings, not softer landings.

Next: keep p1 / p4's table (shift 0.13, held transfer, the IK swing that clears the rug) and work on (1) a longer
stride the policy keeps - stronger stride / speed tracking or training at a fixed higher vx - and (2) the right-side
bias: find whether it is the robot (hip-roll sag, tether, mass) or the policy, before training against it.
