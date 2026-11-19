# Round 24: continuous gaits from the stride tables (2026-11-19 from 00:38, simulation; GPU budget 8 h)

Scripts: `scripts/robot24_common.sh`, `robot24_queue*.sh`. Evaluations: `eval_<run>.txt`, falls: `falls_<run>.txt`.

## Requirement (operator, 2026-11-19)

Produce several policies that fully meet the continuous-gait requirement: no settle gate, no timed hold, no pause
between steps, at the table's cadence and the desired speed. Robustness is secondary: the operator will hold them on
the robot to collect data for the next round. Any clock-rate variation must stay bounded, so it can't bring the
pause back.

## Stage 1: z0 onto the four stride tables, gate off (z0's recipe otherwise)

| run | table | step | both down | clearance |
|---|---|---|---|---|
| s4h_z0 | s4h_relax | 0.5 s | 0.09 s | 30 mm |
| s3h_z0 | s3h_brisk | 0.4 | 0.04 | 25 |
| s4_z0 | s4_relax | 0.5 | 0.12 | 8 |
| s3_z0 | s3_brisk | 0.4 | 0.08 | 6 |

Operator update (00:50): a mix: safer gaits with the gate AND aggressive gate-free (continuous) ones.

s4h_z0: z0 can't jump from 1.2 to 0.5 s steps: 65 steps per walk untrained, 60 at 10M (z0 on its own table ~450).
The s4h table played open loop in the simulator: 0 % up, the board lifts ~0.1 s late and peaks late (servo lag is a
fifth of a 0.42 s swing). Stage 1's other runs (s3h / s4 / s3 from z0) were dropped.

## Stage 2: a cadence ladder (queue2), gated and gate-free branches

New stride tables, 30 mm clearance, the balance point on the boards throughout:

| table | step | both down | sway | peak knee |
|---|---|---|---|---|
| u8h_stride | 0.80 s | 0.24 s | 44 mm | 302 deg/s |
| u65h_stride | 0.65 | 0.15 | 37 | 353 |

| run | table, start | gate |
|---|---|---|
| u8_z0 | u8h, z0 | off |
| u8g_z0 | u8h, z0 | on (z0's, at the table's lift-offs) |
| u65_u8 | u65h, u8_z0 | off |
| u65g_u8g | u65h, u8g_z0 | on |
| s4h_u65 | s4h, u65_u8 | off |

## Stage 3 (01:25): a gait start in the tables

- u8_z0 (0.8 s steps, gate off): 66 -> 93 steps per walk at 10M; still falling in the first 2 s, as s4h_z0 did.
- CPU traces (z0's policy on the new tables, scratchpad fall24.py):
  - forward falls with both feet down in the first second: x_off -0.03 left the standing pose 0.22 rad of hip / knee
    off z0's, the centre of mass 20-30 mm ahead of the board centres;
  - with the offset removed at rest: sideways falls at 1.4-1.8 s, in the first steps.
- The cause: the sway is a pendulum orbit, crossing the centre at ~0.15 m/s sideways at u 0 (mid double support),
  but every start (training, evaluation, the robot) is from rest with the clock at u 0 and the first lift-off 0.1 s
  in. Training never gets past the start, so it can't learn the steady gait.
- New tables (stride_gait.py --ramp-v, --x-off-v, nv 9): sway, lift and the foot offset grow linearly with speed
  from 0 (z0's standing pose) to full at the lowest training command; the command ramp at each start brings the
  first steps in small.

| table | step | both down | full gait from |
|---|---|---|---|
| u8r_stride | 0.80 s | 0.24 s | 0.0225 m/s |
| u65r_stride | 0.65 | 0.16 | 0.0277 |
| s4hr_relax | 0.50 | 0.08 | 0.036 |
| s3hr_brisk | 0.40 | 0.05 | 0.045 |

Queue3: u8r_z0 (gate off), u8rg_z0 (gate on). Failed runs (< 200 steps per walk) get only the fall diagnosis.

## Stage 4 (01:57)

- u8_z0 (unramped, gate off): 66 -> 71 steps per walk; all 48 falls in the first right swing (u 0.21-0.36), right or
  forward, the weight never across.
- u8r_z0 (ramped, gate off): 130 -> 118. The ramp doubles the start, but training doesn't improve it. Falls at
  ~2.0 s (the sway near full by then), in the first full right swings, in every direction. Clearance 26 mm. The legs
  spread 19 deg in hip roll mid-swing (z0's own 1.2 s shift, 13.8 deg at lift-off) against the table's few: z0's
  prior fights the table.
- Joint accelerations aren't the difference: the knee peaks at 158 rad/s^2 in u8r against 101 in z0's c_K2gr (the
  servo ramp is ~35-39).
- Queued: u8rg_z0 (ramped, gate on, running), then u8r_new from scratch (40M, gate off).

## Stage 5 (02:37): training learned to fall

- u8rg_z0 (ramped, gate on): 131 -> 158 steps per walk, 6 % up (79 cm). Falls at the held first lift-off (u 0.08),
  mostly forward with both feet down. The gate freezes the table mid-sway, while the body crosses the centre at
  speed: the gate doesn't suit a pendulum sway. The safer gated policies stay round 23's (z0, g0g, g7g).
- u8r_new (from scratch): 54 -> 46 at 10M; stopped at ~25M.
- **The cause:** per step alive, u8r_new pays tilt_lim -11.9 (body past 4.6 deg, in deg^2) against ~+6 of positive
  terms; z0 on its own gait pays -0.5. Dying early is the best return, so every run learns to fall. Round 23's
  collapses at faster cadences had the same signature (4.0 against 0.65 per step), and falls costing 300 didn't
  offset it.
- The CoM sits 17 mm ahead of the stance board centres in the stride tables at speed (x_off -0.03; c_K2gr -2.5).
  Smaller offsets tilt the boards in double support (no ankle pitch: the pelvis height for a level board depends on
  the foot's x). Kept for now.
- Queue5: tilt_lim -0.05 -> -0.005, falls cost 100: u8r_t (0.8 s), s4hr_t (0.5 s), both from z0, gate off.

## Stage 6 (03:01) and the start problem

- u8r_t (tilt_lim -0.005): the per-step return is now positive (+5), but the walks don't get longer (129 -> 105); in the
  robot-like evaluation it falls at once. s4hr_t stopped at the start.
- Queue6 (running): round 23's continuous-sweep tables with the light tilt penalty, gate off: d4t_t10 (1.0 s steps,
  from z0), then d1t_half (0.825 s, from d4t). They test whether round 23's gate-off collapses were the tilt penalty.
- **The open-loop test** (scratchpad ideal24.py: no hidden offset, no flex, no ramp, no delay, hard floor; the table
  played with zero action):
  - both z0's and the stride tables stand at speed 0;
  - u8r walks forward 89 mm, then falls sideways at 2.2 s. In the first right swing (stance left) the CoM goes to -11
    mm, toward the swinging foot, and every step adds to it.
- **Why:** a pendulum sway needs the body crossing the centre at ~0.15 m/s sideways toward the new stance foot as the
  other lifts (the orbit's energy for a pivot 28.5 mm out). From rest it has none, so it falls away from the stance
  foot. The speed ramp (--ramp-v) can't fix this: a smaller sway doesn't lower the speed needed while the pivot (on
  the board) stays put.
- So a pendulum gait needs a gait start: shift the weight onto the stance foot, lift the swing foot, and join the
  orbit at mid-stance (u 0.25), where the sideways speed is zero. Next: a start prelude in the env (and the robot
  runner).

## Stage 7 (03:08): the gait start, a larger sway

- **The gait start:** `EnvConfig.start_u` / `start_pre` (also `evaluation.CpuBiped`, and `policy_stand.py` reads it
  from the run's config). The clock starts at u 0.25 (mid right swing, where the orbit's sideways speed is zero) and
  holds 0.8 s while the reference blends from the stance pose: the hip / ankle rolls over 0.5 s (the weight onto the
  left foot, both down), then the rest over 0.3 s (the right foot up). stride_gait.py `--inplace`: the zero-speed row
  steps in place with the full sway, so the start has a pose to blend to.
- **Open loop** (ideal simulator, the gait start): the shift reaches a CoM 16 mm toward the stance foot, against the
  table's 22. The board's inner edge is at 15.5, so the board tips over it and the body falls toward the swing foot.
- **Sway:** the pivot moved outward, from 12 mm inside the board centre to 10 mm outside (`--zmp-in -0.01`). Open
  loop, the 0.8 s table then walks for 2.4 s with the start (20 mm outside: 3.2 s, 310 mm, then forward).

| table | step | both down | sway (p-p) | clearance | knee peak |
|---|---|---|---|---|---|
| u8s_stride | 0.80 s | 0.24 s | 77 mm | 29 mm | 311 deg/s |
| s4hs_relax | 0.50 | 0.09 | 51 | 29 | 415 |
| s3hs_brisk | 0.40 | 0.04 | 39 | 23 | 478 |

  The sway is larger than the operator's pick (s4h 29 mm): the robot needs the CoM well over the board in single
  support (no ankle pitch, compliant hip roll).
- Queue7: the d4t_t10 evaluation (round 23's 1.0 s continuous sweep, light tilt penalty; 352 steps per walk
  untrained), then u8s_z0, s4hs_z0 (z0, the gait start, gate off, light tilt penalty).

## Stage 7b (03:17): a gentler fine-tune; gate-free candidates without training

- d4t_t10 (round 23's c_D4t10, z0, gate off, light tilt penalty): 352 -> 205 steps per walk at 10M. Still collapses:
  round 23's collapses weren't the tilt penalty. Next in queue7: d4l_t10, the same with learning rate 5e-5 (3e-4).
- **Gate-free robot candidates, no training:** z0_y2c's weights, unchanged, with round 23's continuous-sweep tables
  and the gate off (runs/z0_d0free, runs/z0_d4free: copied params, edited config). In the simulator (robot settings, 48 walks):

| run | table | up | cm in 20 s | a step every | both down per step |
|---|---|---|---|---|---|
| z0_d0free | c_D0x75 (1.2 s, no stop at the centre) | 71 % | 41 | 1.01 s | 0.70 s |
| z0_d4free | c_D4t10 (1.0 s) | 19 % | 43 | 0.84 | 0.48 |
| *z0_y2c (gated, round 23)* | *c_K2gr* | *96 %* | *39* | | *0.85* |

## Stage 8 (04:17): every gate-off fine-tune degrades

- u8s_z0 (the gait start, sway pivot 10 mm out): 103 -> 68 -> 62 steps per walk; most falls during the 0.8 s start
  (u 0.25, the clock held), sideways.
- d4t_t10 finished 352 -> 205 -> 126; its evaluation return fell with it (2016 -> 745): it isn't reward hacking.
- d4l_t10 (learning rate 5e-5): 357 -> 143 -> 102, faster: not the step size.
- s4hs_z0 stopped at the start.
- The pattern across rounds 23-24: every gate-off fine-tune of z0 degrades, even on a table z0 walks untrained
  (g0_ctrl on c_D0x75: 71 % up untrained -> 54-62 %; g0b on from it: 421 -> 272), while gated runs improve (g0g).
- Hypothesis: exploration noise. The training policy's action std is ~0.6 (x action scale 0.25 = ~0.15 rad on every
  joint, every tick); the gate absorbs the wobble it causes, and without it the training walks fall early.
- Test (queue8): z0's params with the noise scale lowered to ~0.25 (runs/z0_lowstd; the scale bias -1.06):
  - n0_lowstd: c_D0x75, g0_ctrl's recipe (gate off, falls 300);
  - n4_lowstd: c_D4t10, d4t's recipe.

## Stage 9 (04:40): train gated, deploy gate-free

- n0_lowstd (noise scale ~0.25): 443 -> 376 at 10M, worse than g0_ctrl (437 -> 422): the noise isn't the cause.
  queue8's n4_lowstd dropped.
- **Gated policies walk without the gate** (simulator, robot settings, 48 walks, the gate off at evaluation):

| run | table | up | cm in 20 s | a step every | both down per step |
|---|---|---|---|---|---|
| g0g_gate | c_D0x75 (1.2 s, no stop) | 96 % (gated 85-90) | 42 | 1.20 s | 0.85 s |
| g7g_t11 | c_D7t11 (1.1 s) | 52 % (gated 35-62) | 44 | 1.00 | 0.64 |

- So: train with the gate (training is stable), deploy without it.
- Queue9:
  - gate-free evaluations (results/robot24/free_*.txt) of g0g and g7g;
  - h4g_t10: g7g onto c_D4t10 (1.0 s), gated;
  - u8sg_z0: z0 onto u8s_stride with the gait start, gated at mid-stance (u 0.25 / 0.75, the attitude gate off);
  - h1g_half: h4g onto c_D1h (0.825 s), gated;
  - each evaluated gate-free.
- Gate-free evaluations (free_*.txt, the k5-equivalent commands low / mid / high):
  - g0g_gate: 90 / 98 / 92 % up (34 / 42 / 48 cm), a step every 1.15-1.18 s, both down 0.81-0.84 s per step;
  - g7g_t11: 67 / 50 / 31 % up, a step every 0.97-1.05 s, both down 0.61-0.69 s.
- n0_lowstd finished at 429 steps per walk (443 -> 376 -> 429): no better than g0_ctrl.
- h4g_t10 (g7g onto c_D4t10 1.0 s, gated): held in training, 384 -> 384 -> 371 steps per walk. Gate-free: 23 / 19 / 2 %
  up, a step every 0.87-0.95 s, both down 0.45-0.57 s per step. Falls (38 of 48, median 6.2 s): sideways with a foot
  up, 33 of 38 toward the stance foot (left with the right up, right with the left up). The body overshoots its
  shift: z0's lineage learned a large shift (legs 13.8 deg apart at lift-off) for 1.2 s steps, too much at 1.0 s.

## Stage 10 (05:37): a smaller shift at faster cadences

- u8sg_z0 (stride + the gait start, gated at mid-stance): 97 -> 69 at 10M. The stride tables don't train from z0,
  gated or not.
- After h4g's overshoot: round 23's tables with the shift 0.12 -> 0.08 (legs 9.2 deg apart at lift-off against
  13.8): c_D4s08 (1.0 s), c_D1s08 (0.825 s). Queue10: h4s_t10 (g7g -> c_D4s08, gated), then h1s_half (h4s ->
  c_D1s08, gated), each evaluated gate-free. h1g_half (queue9) dropped.
- u8sg_z0 ended 61 steps per walk; gate-free 0 % up, falls at the gait start (u 0.25). The stride line is closed for
  this round.
- h4s_t10 (smaller shift, 1.0 s, gated): 332 -> 391 at 10M, the first faster-cadence run this round that improves in
  training.
- **h4s_t10** (g7g onto c_D4s08, the shift 0.08, gated): 332 -> 391 -> 433 steps per walk in training. Gate-free
  (free_h4s_t10.txt): 48 / 42 / 33 % up (h4g with the shift 0.12: 23 / 19 / 2), a step every 0.78-0.86 s, both down
  0.44-0.54 s per step (z0 0.85). Falls (25 of 48, median 8.1 s), all with the right foot up: 13 toward the stance
  foot (left, at the right lift-off), 10 toward the swinging foot (right, mid-swing). The overshoot is halved.
- Queue11 (after h1s_half): h4s2_t10 (h4s +20M), h4x_t10 (g7g onto c_D4s06, the shift 0.06 = 6.9 deg).
- h1s_half (h4s onto c_D1s08: 0.825 s, the shift 0.08, gated): 363 -> 373 in training; gate-free only 2 / 2 / 4 % up
  (a step every 0.66-0.69 s, both down 0.29-0.32 s per step). Falls (46): 26 toward the swinging foot mid-swing (u
  0.24-0.33), 14 toward the stance foot at the lift-off: too little shift for 0.825 s.
- h4s2_t10 (h4s +20M): 416 -> 414 at 10M: a plateau.
- Queue12 (the last): h9s_t09, h4s2 onto c_D9s09 (0.9 s steps, the shift 0.09 = 10.3 deg), gated, evaluated
  gate-free. Ends ~07:55.
- **h4s2_t10** (h4s +20M, c_D4s08, gated): 416 -> 414 -> 466 steps per walk. Gate-free: **75 / 81 / 75 % up**
  (40 / 47 / 53 cm), a step every 0.82-0.85 s, both down 0.52-0.53 s per step (z0 0.85). Falls only 11 of 48
  (median 12 s): 5 left at the right lift-off, 3 right mid-swing, 3 back. The best gate-free candidate of the round.
- h4x_t10 (g7g onto c_D4s06, the shift 0.06): 268 -> 436 at 10M.
- h4x_t10 (g7g onto c_D4s06, the shift 0.06 = 6.9 deg, gated): 268 -> 436 -> 540 in training. Gate-free 75 / 46 / 33 %
  up, a step every 0.88-0.93 s, both down 0.53-0.61 s. Falls 26 (median 10.5 s): 21 to the left with the right foot
  up, around the right lift-off (u 0.91-0.20): the robot's left lean, now the main failure.
- h9s_t09 (h4s2 onto c_D9s09, 0.9 s): 403 untrained. Queue13 (the last slot, ends ~08:30): h4x2_t10 (h4x +20M).
- h9s_t09 (h4s2 onto c_D9s09: 0.9 s, the shift 0.09, gated): 403 -> 402 -> 433. Gate-free 69 / 44 / 12 % up, a step
  every 0.77-0.83 s, both down 0.39-0.48 s per step. Falls 31 (median 7.6 s): 19 to the right mid-swing with the right
  foot up (too little shift for 0.9 s, as at 0.825), 10 left at the right lift-off.
- h4x2_t10 (h4x +20M, the shift 0.06): 555 -> 503 -> 583. Gate-free 88 / 77 / 71 % up but short (21 / 29 / 37 cm),
  a step every 0.92-0.95 s, both down 0.58-0.61 s per step. Falls only 8 of 48, 7 to the left with the right foot up.

## Summary (budget ended 08:38)

1. **The stride tables (the operator's 3 and 4) didn't train.** In order, the causes found and fixed:
   - x_off -0.03 left the CoM 17-25 mm ahead of the boards;
   - training learned to fall: the tilt penalty outweighed every reward;
   - a start from rest can't give the pendulum sway's sideways speed: the gait start (`start_u`);
   - the sway was too small for this robot's compliant hip roll: the pivot moved outward.
   Even so, z0 fine-tuned onto them degraded every time, gated or not, from z0 or from scratch. Open loop in the
   ideal simulator the best one walks 2-3 s. The code stays (stride_gait.py, start_u in the env / CPU sim / runner).
2. **Every gate-off fine-tune degrades; gated ones improve.** Not the tilt penalty, the learning rate, or the
   exploration noise (each tested). Not explained yet.
3. **Train gated, run gate-free.** Policies trained with the gate walk without it, continuously at the table's
   timing (g0g: 96 % gate-free against 85-90 % gated).
4. **At faster cadences, a smaller weight shift.** Round 23's quasi-static tables at 1.0 s steps with z0's shift
   overshoot toward the stance foot; the shift 0.08 (9.2 deg) halves the falls. At 0.9 / 0.825 s, 0.09 / 0.08 is too
   little (falls toward the swinging foot mid-swing).

## For the robot

`--no-gate` (new in policy_stand.py) runs a gated-trained policy gate-free. From `hardware/leg_p0/software`, all
`--enable --act-lpf 0.7 --video`. The command is k5's step length at each cadence: 0.02 x 1.2 / T_step.

Gate-free (continuous, the operator's hand close):

| run | command | step | gate-free up (sim, mid command) | both down per step | note |
|---|---|---|---|---|---|
| h4s2_t10 | `--run h4s2_t10 --no-gate --vx 0.024` | 1.0 s table, a step every 0.84 s | 81 % | 0.53 s | the best of the round |
| h4x2_t10 | `--run h4x2_t10 --no-gate --vx 0.024` | 1.0 s, 0.95 s | 77 % | 0.61 | smallest shift, short steps |
| h9s_t09 | `--run h9s_t09 --no-gate --vx 0.0267` | 0.9 s, 0.80 s | 44 % | 0.45 | the fastest that walks; hold it |
| g0g_gate | `--run g0g_gate --no-gate --vx 0.02` | 1.2 s, 1.18 s | 98 % | 0.84 | z0's cadence, no pause at the centre |

Gated (safer): z0_y2c, g0g_gate, g7g_t11 (round 23), and h4s2_t10 without `--no-gate`.

Baseline in the same session: `--run z0_y2c --vx 0.02`.

The gait-start code in policy_stand.py (`start_u`) is only used by the stride runs (none are robot candidates);
it has not been run on the robot.

## Robot trials, 2026-11-19 10:25–10:33 (20 logs)

| run | trials | steps before fall/stop | how they ended |
|---|---|---|---|
| h4s2_t10 (gate off ×6, on ×2) | 8 | 10, 10, 5, 2, 8, 9, 12, 5 | 4 fell right, 4 stopped by hand |
| h4x2_t10 gate off | 2 | 10, 7 | right, left |
| h9s_t09 gate off | 2 | 8, 6 | stopped, right |
| g0g_gate gate off | 3 | 6, 8, 10 | all right |
| g7g_t11 gated | 2 | 2, 10 | forward, stopped |
| k5_high2 | 1 | 9 | left |
| z0_y2c gated | 2 | 15, 17 | forward, right |

The gated runs barely held the clock: 0.1–0.8 s in a whole trial. Standing roll before the start was unchanged from 11-18 (−1 to −3°). Voltage 5.2 V.

**How the falls happen.**
- 12 of 15 went right, mostly with pitch forward, starting early in the left stance (u 0.2–0.35, the right foot swinging).
- They take 0.5–1.4 s to grow from 5° to 20° of tilt. That is a slow fall, with time for one or two corrective steps.

**The policy's balance feedback, measured on the robot.**
- To isolate the feedback, I subtracted each signal's mean at the same gait phase. What remains:
  - ankle roll: about 0.3° per degree of body roll (R² 0.3–0.5);
  - hip roll: about 0.1° per degree (R² ≈ 0.1).
- **No catching step.** At touchdown the swing foot's sideways position relative to the pelvis moves −4 to −8 mm per degree of lean (r −0.6 to −0.9).
  - That matches the whole robot tipping as one rigid piece (about −5 mm/deg at the pelvis height).
  - The foot just rides along with the lean. A catching step would move it the other way, by about +4 mm/deg plus a speed term.
  - Every run shows this, gated or not.
- So each step lands where the table puts it, whatever the body is doing. The policy only corrects with a weak ankle torque, and the clock never waits or hurries. This is the "commit to the step and hope" the operator sees.
- Scripts are in the scratchpad: r21.py, r24b.py, fb24b.py, place24b.py.

## Robot vs simulator, same policies (results/robot24/robot_vs_sim_1119.txt)

- **Setup.** GPU "sim robot" (gait_eval's settings), 32 walkers × 20 s, against the 11-19 robot logs. The first 1 s and the last 1 s before a fall are left out.
- **Joints: the same.**
  - Per-cycle joint motion matches within 1–2° (p2p), and so does the off-rhythm joint noise.
  - Mean joint angles agree within 1–2°, except the knee: about 3° straighter on the robot.
- **Body: different.**
  - **Sideways.** The robot rolls right as far as the sim does (+6.5 to +11.7° per cycle against +4 to +7.5°). It rolls back left only half as far: −2.4 to −5° against −6 to −8.5°.
    - Late in the cycle (u 0.6–0.9), the sim swings to −4 to −7°. The robot (h4s2, g0g, h9s) stays at +2 to +4°.
    - So the body arrives late and short over the left foot. The right foot then lifts with the weight still to the right, and 12 of 15 falls go right during the right swing.
    - z0 comes closest (−3.4° at u 0.9) and lasts longest.
  - **Fore-aft.** The robot rocks forward and back twice a cycle, 6–9° p2p against 2–5° in the sim. Pitch rate rms is 18–27 °/s against 12–13. Roll rate rms is 23–25 °/s against 17–19.
- **Conclusion.** The policy does the same thing in both. The difference is between the encoders and the body: give the encoders can't see (gearbox, frame, the hip roll under load), the boards rocking or slipping, or the mass distribution. It is lopsided, rolling right freely but not left.

## Does the policy react to the robot's different tilt? (replay of the robot logs, scratchpad replay24.py)

- **Check.** Rebuilt from the logs, the policy's inputs reproduce the logged actions: median error 0.016 action units, about 0.2°.
- **The swap.** The simulator's mean body tilt at the same gait phase replaced the robot's (gravity vector, then the gyro too). Everything else stayed the robot's. The tilt gap is 3.9° rms (h4s2_t10) and 2.9° (z0_y2c).
- **Command change per joint (rms):**

  | | per tick (through the 0.7 action filter) | held for 5+ ticks (÷0.3) |
  |---|---|---|
  | h4s2_t10, tilt only | 0.4–0.7° | about 1.3–2.4° |
  | h4s2_t10, tilt + gyro | 0.5–1.1° | up to about 3.5° (knees) |
  | z0_y2c, tilt only | 0.2–0.4° | about 0.7–1.3° |

- **Against the rhythm**, the joints swing 12–40° per cycle (hip roll 12–21°), set by the table and the gait clock.
- **So the policy sees the different body and changes its commands by about 1–3°.** That is too little to bring the body back to the sideways swing it was trained on. What it does is mostly set by the clock.

## How far each step strays from the trained path (scratchpad miss24.py)

- **Measure.** Per step (half a cycle): the worst tilt away from the sim's mean path at the same gait phase. Sim steps are from the "sim robot" (no pushes, like training). Robot steps are from the 11-19 logs.

| run | sim p50 / p90 / p99 | robot ordinary steps p50 / p90 | robot: the step before the falling step |
|---|---|---|---|
| h4s2_t10 | 4.4 / 7.1 / 12.3 | 9.6 / 14.1 | 6–16 (median 8.8) |
| z0_y2c | 4.0 / 6.4 / 10.5 | 5.9 / 8.8 | 11–16 |
| g0g_gate | 4.4 / 7.0 / 12.2 | 10.8 / 15.7 | 9–10 |
| h9s_t09 | 4.2 / 7.4 / 15.1 | 8.2 / 15.7 | 9.3 |

- **The robot's typical step is as far off the path as the sim's worst ~10 %.**
  - In the sim, steps 9–12° off are mostly recovered: falls within 2 steps 0 % (h4s2, g0g), 9 % (z0), 19 % (h9s).
  - On the robot, a step at that level is followed by a fall.
  - The sim's rare 12°+ steps fall 43–89 % of the time, except g0g (0 of 6).
- **The training config** (h4s2_t10) has no pushes (push_vel 0, push_dv [], tug 0), no ring, dr_com 0 and reset_tilt 0.04. Training rarely sees a body far off its path.
- **E6** (ex_shift_full_L, 2026-11-13; the robot tips over the left board after 3° of hip-roll shift) **is still not reproduced.** exercises_sim on the m2 model, rigid or with a plain hip-roll give of k 10 or 4: never tips, roll ≤ 0.3° through 8° of shift. E6 exists for the left side only.

## E6 weight shift, both sides (2026-11-19 11:20–11:23; logs hardware/leg_p0/software/ex_shift*.csv)

| run | onto | tips at hip-roll shift | pelvis roll just before | far knee load |
|---|---|---|---|---|
| ex_shiftR_20261119_112011 | R | 5.8° | +5.2° at 5.5° | unchanged until the tip |
| ex_shift_full_R_2 | R | 6.0° | +5.6° | unchanged until 5.8°, then gone in 0.5 s |
| ex_shift_full_L_2 | L | 4.0° | −4.2° | drops slowly from 3° (40 → 24), gone at the tip |
| ex_shift_full_L (11-13) | L | 3.0° | −3.1° | unchanged until the tip |

- **About 2° lopsided:** onto the left board it tips at 3–4°, onto the right at 6°.
- **Every run shows the same shape:**
  - about 0.3–0.5° of pelvis roll per degree of shift at first;
  - then it runs away within 0.5 s, before the far board has unloaded;
  - so the robot tips over its give, never standing on one board.
- **The sim (m2, rigid or with the plain give)** never tips through 8° and rolls ≤ 0.3°.
- **Caveat:** the 112011 run stood 4.6° off its usual standing roll before the ramp (roll −5.4 against −0.8 to −1.0), yet tipped at the same 5.8°.

## E6 again, cable lifted (11:32–11:33; these overwrote the earlier ex_shift_full_{R,L}_2 files)

| | standing roll | tips at shift: R / L | lean at the runaway: R / L |
|---|---|---|---|
| cable resting (11:23 runs, now overwritten) | −0.8 / −1.0 | 6.0 / 4.0 | 5.6 / 4.2 |
| cable lifted | −0.3 / −0.6 | 5.8 / 4.8 | 4.7 / 4.1 |

- **The cable accounts for about half the lopsidedness.** Lifted, the standing lean halves and the left tip moves from 4.0° to 4.8°. The right–left gap drops from 2° to 1°.
- **About 1° remains**, from one run per side, so within the scatter of a single run. It needs repeats before it counts as fixed.
- **Lean per degree of shift** is the same on both sides until the runaway (about 0.4–0.6°/° from 3° on). The give looks symmetric.

## E6 after re-zeroing the left hip roll and tightening the loose left-ankle screws (11:45–11:46, cable lifted)

| | onto | standing roll | tips at shift | lean at the runaway | lean at 3 / 5° of shift | far knee load at the runaway |
|---|---|---|---|---|---|---|
| ex_shift_full_L_4 | L | −0.1 | **7.9** (end of the ramp) | 5.9 | 0.3 / 0.7 | 64 → 16: **the far board unloaded** |
| ex_shift_full_L_2 (overwritten again; really `--side R`) | R | +0.3 | 6.2–6.4 | 4.6 | 0.4 / 1.8 | −112 → −52 |

- **Left:** from 3–4.8° to 7.9°. Its give is far stiffer: 0.7° of lean at 5° of shift, against 2.4° at 4° before.
- **First time** a board unloaded before the tip.
- **The right side** is unchanged (5.8–6.4°) and now the softer side. Check the right-ankle screws the same way.
- **Standing roll** is about 0 (−0.1 / +0.3).
