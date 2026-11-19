# Round 23: less time on both feet (2026-11-18 12:14-19:14, simulation)

Scripts: `scripts/robot23_common.sh`, `robot23_queue.sh` .. `robot23_queue8.sh` (the plan changed with each finding).
Evaluations: `eval_<run>.txt`; fall diagnostics: `falls_<run>.txt` (`gait_eval.py --falls`, new).

## Why

Robot round 22: z0_y2c walks with k5's steps and less lean and shake than k5. The operator's next goal is continuity:
- no visible wait between steps;
- the rear foot lifting as the front one lands;
- **the target metric: the time with both feet down.**
  - It is 0.75 s of every 1.2 s step in z0's table, and 0.85 s measured in the simulator.
  - Halve it, and try a near-zero one.
- Soft landings are a stretch goal only.
- When a gait fails, find out why (which instability) before tuning.

## New tools

- **`shift_gait.py step --xfer`:** the sideways weight transfer as one continuous sweep, centred on the double support.
  - It passes the centre at speed. k5 / z0 stop the body at the centre between the release and the next shift.
  - A sweep longer than the double support starts before touchdown and ends after lift-off (a dynamic sway).
  - The clock starts mid-sweep, so the standing pose is level.
- **`gait_eval.py`:**
  - now reports both feet down (% of the time, s per step, from the simulated contacts);
  - `--falls` gives, per fall, when the tilt passed 10 deg (gait phase), which way, which foot was up, the sideways rock
    per cycle before it, and the body's shake spectrum.

## The tables

All keep z0's swing: 0.45 s, lift 0.75 / 0.75, 30 mm, a continuous pelvis. Faster cadences scale the command range
by 1.2 / T_step, so each step stays as long as k5's at the k5-equivalent command.

| table | step | both down | sweep | legs apart at lift-off |
|---|---|---|---|---|
| c_K2gr (z0) | 1.2 s | 0.75 s | stop at the centre | 13.8 deg |
| c_D0x75 | 1.2 | 0.75 | 0.75 s, no stop | 13.8 |
| c_D7t11 | 1.1 | 0.65 | 0.65 | 13.8 |
| c_D4t10 | 1.0 | 0.55 | 0.55 | 13.8 |
| c_D1h | 0.825 | 0.375 (half) | 0.375 | 13.8 |
| c_D2hd | 0.825 | 0.375 | 0.55 (dynamic) | 12.3 |
| c_D6a | 0.6 | 0.15 | 0.30 | ~12 |
| c_D3a | 0.6 | 0.15 | 0.45 (dynamic) | 6.9 |
| c_D5a | 0.55 | 0.10 | 0.50 (dynamic) | 4.4 |

The fastest hip-roll movement stays at z0's (39-58 deg/s against 43), because the stop at the centre is gone.

## Results (simulation)

Columns:
- up % / cm in 20 s at the k5-equivalent commands (0.015 / 0.025 / 0.035 x 1.2 / T_step);
- loose and slow hips at the middle one;
- through shoves 0-0.1 m/s;
- both feet down at the middle command (s per step).

"Recipe": z0's is the gate on and falls costing 20. "Off / 300" is the gate off and falls costing 300.

| run | table, start, recipe | up / cm | loose | slow | shove | both down |
|---|---|---|---|---|---|---|
| *z0_y2c* | *c_K2gr* | *94 % 34 · 96 % 39 · 96 % 42* | *29 %* | *60 %* | *98 %* | *0.85* |
| d1_half | D1h 0.825 s, k5, z0's | 2 % · 2 % · 0 % | 0 | 0 | 1 % | 0.47 |
| d3_amb | D3a 0.6 s, k5, z0's | 0 % | 0 | 0 | 0 | 0.59 |
| e1_half | D1h, k5, gate off | 0 % | 0 | 0 | 0 | 0.29 |
| f1_half | D1h, k5, off / 300 | 0 % | 0 | 0 | 0 | 0.22 |
| f4_t10 | D4t10 1.0 s, z0, off / 300 | 0 % | 0 | 0 | 0 | 0.78 |
| g1_t10q1 | D4t10, z0, off / 300, ref_q 1 | 0 % | 0 | 0 | 0 | 0.98 |
| g6_t10slow | D4t10 at k5's speeds, z0, off / 300 | 0 % | 0 | 0 | 2 % | 0.62 |
| g0_ctrl | D0x75 1.2 s, z0, off / 300 | 54 % 42 · 62 % 46 · 50 % 48 | 10 % | 42 % | 66 % | 0.69 |
| g0b_ctrl | D0x75, g0 + 20M, off / 300 | 19 % · 8 % · 10 % | 2 % | 4 % | 20 % | 0.57 |
| **g0g_gate** | D0x75, z0, z0's | 90 % 33 · 90 % 42 · 85 % 47 | 21 % | 44 % | 92 % | 0.82 |
| **g7g_t11** | D7t11 1.1 s, g0g, z0's | 62 % 39 · 52 % 42 · 35 % 45 | 0 % | 27 % | 63 % | **0.67** |
| g4r_t10 | D4t10 1.0 s, g0g, z0's, a relaxed gate (0.6 rad/s, at most 0.3 s) | 6 % · 4 % · 2 % | 0 % | 4 % | 3 % | 0.56 |

g7g also lands softer: 230 mm/s against z0's 277.

## Why the faster cadences fail (simulation)

1. **The settle gate stalls the clock.**
   - The gate holds the next lift-off until the body turns slower than 0.3 rad/s.
   - At 0.825 / 0.6 s steps the body never gets that slow: the clock is held 38 % (d1) and 66 % (d3) of the time,
     which puts the wait back between steps.
   - Most falls start at a held lift-off (d1: 9 of 15 at u 0.10-0.11 / 0.61; d3: 23 at u 0.55).
2. **Without the gate, training degrades.**
   - Every run trained with the gate off got worse as training went on:

     | run | steps per walk (start -> end) | body past 4.6 deg (tilt penalty per step) |
     |---|---|---|
     | e1 | 275 -> 106 | |
     | f1 | 285 -> 129 | |
     | f4 | 369 -> 232 | |
     | g1 | 371 -> 178 | |
     | g6 | 394 -> 210 | |
     | g0b | 421 -> 272 | -0.99 -> -1.63 |

   - The same no-stop table trained with the gate on improved (g0g 428 -> 502, -0.83 -> -0.59), as z0's lineage does.
3. **Ruled out:**
   - Copying the table: ref_q 3 -> 1 (g1) still collapsed.
   - Forward speed: k5's speeds at 1.0 s steps (g6), and low commands at evaluation, still fell.
   - Cheap falls: falls costing 300 (f-runs) still collapsed.
   - The clock start: the command ramps from 0, so the steps grow gently.
   - The no-stop sweep itself: with the gate on (g0g) it is as robust as z0 and a bit farther at speed.
4. **The falls of the collapsed policies:**
   - in the first 2-3 s, mostly with one foot up;
   - g1: 43 of 48 in the first right swing (u 0.24-0.36), 28 forward and 15 left: the body topples before the
     swinging foot lands;
   - the open-loop tables fall at once (the policies' standing corrections are needed), so that test doesn't
     separate tables.
5. **Untrained beats trained at faster cadences.** z0 walks c_D4t10 at 31 % untrained, and 0 % after 25M of training
   on it (f4). Round 18 saw the same with n7 at 1.0 s.

So the conflict is the settle gate: training needs it, and at faster cadences it stops the gait. g7g (1.1 s steps,
gate on, from g0g) is the first faster cadence that trains without collapsing (500 -> 453 steps per walk). It cuts
the double support by 21 % (0.67 against 0.85 s) and lands softer, but is not yet robust (35-62 % up).

g4r: the relaxed gate didn't hold the 1.0 s cadence (460 -> 234 steps per walk in training). 34 of its 47 falls
went right, most during the right swing (u 0.2-0.32).

g7g's falls (23 of 48, from a median 8.8 s): 13 to the left, most in the late left swing or just after the left
touchdown (u 0.75-0.96; touchdown at 0.85). The body keeps going over the landing foot. It is the same side the robot
leans toward.

## For the robot

All `--enable --act-lpf 0.7 --video`, from `hardware/leg_p0/software`. The runner reads each run's step time and
gate from its config. The k5-equivalent command is 0.02 x 1.2 / T_step.

1. `python policy_stand.py run --run g0g_gate --enable --vx 0.02 --act-lpf 0.7 --video`
   - z0's steps with the no-stop sideways sweep, 1.2 s steps;
   - in simulation 85-90 % up against z0's 94-96 %, a little farther at speed.
   - Watch: does the body flow from step to step without the pause at the centre?
2. `python policy_stand.py run --run g7g_t11 --enable --vx 0.022 --act-lpf 0.7 --video`
   - 1.1 s steps: both feet down 21 % less than z0, softer landings in simulation;
   - only 35-62 % up in simulation, so keep a hand close. Its falls are to the left, over the landing foot.
3. Baseline, same session: `python policy_stand.py run --run z0_y2c --enable --vx 0.02 --act-lpf 0.7 --video`.

Compare (repo root, today's logs from 19:00 on):
`.venv/bin/python scripts/robot19_analysis/gait_metrics.py $(for f in hardware/leg_p0/software/stand_{g0g_gate,g7g_t11,z0_y2c}_2026111[89]_*.csv; do case $f in *_frames.csv) ;; *) [ -f "$f" ] && [ $(wc -l <"$f") -gt 200 ] && echo "$f";; esac; done)`
