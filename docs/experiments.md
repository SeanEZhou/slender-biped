# Experiment log

Everything that ran is listed here, failures included.
- E0–E3: model building and audits on CPU, 2026-10-22.
- E4–E22: GPU training, 2026-10-23.

The README has the headline results; this log has the detail behind them.

**How to read the numbers.**
- Unless stated otherwise, results come from the independent CPU evaluator (`scripts/evaluate.py`)
  on held-out seeds 10000+, with the full contact set and a deterministic policy.
- "Alive @30 s" is the share of episodes without a fall.
- A "step" is a touchdown after ≥ 60 ms of air time; shorter airborne blips are counted as bounces.
- Torque is reported as RMS in N·m, as a fraction of stall (`rms_tau_frac`), and as a fraction of
  vendor rated torque (`rms_over_rated`).
- Raw per-episode results are in `results/eval/<run>__<test>.json`.

## Run index

| # | run (`runs/…`) | what | outcome |
|---|---|---|---|
| E5 | `stand_A8_nodr_s0` | stand, quiet-stand reward | shuffles; 33 % alive @30 s; over rated torque |
| E6 | `stand_A8_step_nodr_s0` | stand v1, stepping in place | 96 % @30 s; 1.29 × rated torque |
| E7 | `stand_B8_step_nodr_s0` | v1, rocker B | 85 % @30 s; transfers worse |
| E8 | `stand_A8_v2_nodr_s0` | stand v2, hardware penalties | **99 % @30 s, ≤ 0.88 × rated** |
| E9 | `walk_A8_dr_s0` | walk v2 | steps in place; stopped at 60 M |
| E10 | `walk_A8_v3_dr_s0` | walk v3 | walks, drifts 80° off heading |
| E11 | `walk_A8_v4_ft` | walk v4 fine-tune | regression |
| E12 | `stand_B8_v2_nodr_s0` | stand v2, rocker B | ties A |
| E13 | `stand_A8_v2_dr_s0` | stand v2 + DR fine-tune | no robustness gain |
| E14 | `walk_A8_v5_dr_s0` | walk v5, heading reward | walks straight; can't stop; regresses late |
| E15 | `walk_B8_v5_dr_s0` | walk v5, rocker B | ties A |
| E16 | `walk_A8_v6_ft` | stop fine-tune | collapsed (critic re-initialized) |
| E17 | `walk_A8_v6b_ft` | stop fine-tune, critic restored | **all gates pass (selected policy)** |
| E18 | `stand_still_A8_ft`, `stand_still_B8_ft` | quiet-stand fine-tunes | never stationary |
| E19 | `stand_A8_v2strict_nodr_s0` | 0.25 × stall cap | 95 % @30 s |
| E20 | `stand_slimA8_v2_nodr_s0` | slim Dynamixel hips | 91 % @30 s; speed-limited |
| E21 | `stand_A8_v2_robust_s0` | noise + pushes from scratch | noise fixed; pushes not |
| E22 | `stand_A8_v2_nodr_s1` | seed 1 | 97 % vs 99 %: reproducible |

## E0 — environment
- uv venv with Python 3.10: mujoco 3.17.0, torch 2.14.1+cpu, stable-baselines3 2.9.0.
- The GPU was busy with another job, so everything here ran on CPU. Offscreen rendering
  is forced onto Mesa llvmpipe (`slender_biped/render.py`).
- OSMesa and EGL offscreen both failed on this WSL2 host. GLFW + llvmpipe works.

## E1 — Gate A packaging iterations (`scripts/inspect_model.py`)
| # | change | result |
|---|---|---|
| 1 | first build: elliptical thigh 42 mm, shin 20 mm, servos centred | knee servo protrudes 3.2 mm, ankle 2.2 mm (**fail**) |
| 2 | packaging check now requires 1.2 mm wall clearance | hip servos also fail (roll +0.26 mm, yaw −0.06 mm) |
| 3 | sweeps: thigh superellipse n, width, servo z | linear taper can't hold a 35 mm case; needs a mid-section |
| 4 | piecewise tube (top/mid/bottom); thigh 44 mm n = 4; shin 22 × 36 n = 3.5 | knee and ankle pass |
| 5 | roll-cover inset 4.5 mm; hip cover 62 × 47 × 36; pelvis n = 3.8 | all baseline pass; pelvis boxy (66 mm deep) |
| 6 | yaw servos point inward; pelvis 54 mm deep, n = 3.4 | pass; rounder, shallower pelvis |
| — | direct_knee variant | knee case 1.0 mm short of the required clearance (flagged) |
| — | slim_dynamixel variant | yaw case touches the shell (+0.04 mm) (flagged) |

## E2 — Gate B torque audit (`scripts/torque_audit.py`)
- Initial bug: `mj_forward` overwrote the prescribed `qacc`, so all static torques came out 0. Fixed.
- First pass with the 12 V STS3215-C018 as baseline: the knee can follow swing only at
  ≤ 1 step/s (speed-limited). **Baseline switched to STS3215-C046 (7.4 V, 1:147)**, which has the same case.
- First pass applied one armature (0.004) to every servo. Reflected inertia scales with N²,
  so this overstated C046 knee swing torque about 7×. Now per-servo J_rotor·N² (estimate).
- Deep-crouch rig pose commanded the ankle to 50°, beyond its 30° stop, and saturated it. The pose generator now clamps.

## E3 — Gate C sanity (free base, no support) (`scripts/sanity_sim.py`)
4 s, seed 0, baseline servo set. Fall = pelvis below 60 % of stand height, or tilt > 45°.

| model | limp (back-EMF only) | naive position hold | random targets |
|---|---|---|---|
| A 10-DoF | falls 0.80 s | falls 0.83 s | falls 0.79 s |
| B 10-DoF | falls 1.08 s | falls 0.86 s | falls 0.80 s |
| A 8-DoF | falls 0.74 s | falls 0.76 s | falls 0.77 s |
| B 8-DoF | falls 0.85 s | falls 0.79 s | falls 0.81 s |

As expected, neither foot stands under a non-balancing position hold. The 32 mm rocker
adds only ~0.1–0.3 s. Peak |τ|/stall over the run (including the fall and ground impact)
reached 0.2–1.0. Several runs hit the current limit during the fall. That is logged rather
than hidden, and is not counted as a stance-feasibility result. This is the bar the Phase 1
policy has to clear.
(Rerun after the per-servo armature fix. The first run with armature = 0.004 gave fall times
within ±0.13 s of these.)

## E4 — GPU training infrastructure (2026-10-23)
- The user selected **baseline A (ball tip), 8 DoF** as the preferred form factor. Other configurations are
  explored as comparisons. The GPU (RTX 4070 SUPER) was handed over, with one job at a time.
- Moved to Python 3.12 + JAX 0.12 + MJX 3.17 + Brax 0.14.2. The system CUDA 12.8 on `LD_LIBRARY_PATH`
  shadowed the pip CUDA libs, so `scripts/gpu_python.sh` unsets it. Brax still calls
  `jax.device_put_replicated`, which JAX 0.12 removed; `scripts/train.py` adds a single-GPU shim.
- MJX limits that changed the model:
  - No ellipsoid-box collision, so collision boxes were replaced with capsules.
  - No condim-1 pairs with elliptic cones, so leg-leg pairs use condim 3.
  - All contacts are now **explicit pairs**. The full set (27 pairs) is used for CPU evaluation; a lean
    set (tip-floor plus 5 leg-leg pairs) is used for training. In training, falls are detected
    kinematically and end the episode at that moment either way.
- Throughput (4096 envs, 4 solver iterations, 10 × 2 ms substeps per 20 ms control step):
  - full contact set: 23k control steps/s;
  - lean set: 87k raw; about 13k with PPO, randomization and the servo model at 1024 envs.
- The NumPy policy re-implementation matches Brax inference exactly (max |Δa| = 0.0).

## E5 — Standing, attempt 0: quiet-stand reward (`stand_A8_nodr_s0`, task `stand_quiet`)
- A 8-DoF, no randomization (deterministic baseline), 60 M steps, γ = 0.97, termination −20.
- Reward: upright, height, zero-velocity tracking, both-feet-down bonus (+0.3), touchdown penalty (−0.2),
  plus the regularizers. The exact weights are in `runs/stand_A8_nodr_s0/config.json`.
- At 30 M steps the episode length had plateaued at ~1.6 s, against a passive fall time of ~0.8 s. The
  dominant penalties were body angular velocity and tip slip. The policy learned to *slow its fall*,
  and was discouraged from the stepping a point-footed robot needs. The 10 M checkpoint on CPU:
  median survival 1.8 s, about 10 touchdowns/s, which is chattering rather than stepping.
- Diagnosis:
  - The reward fights the physics. With a ball tip and no ankle, the support is two points, so the
    robot must step to stay up.
  - γ = 0.97 at 50 Hz looks ahead only ~0.7 s.
  - A −20 termination penalty dominates the per-step rewards (~0.4 per step).
- Next, attempt 1 (task `stand`): stepping in place on the gait clock at zero command, γ = 0.99,
  termination −5, alive +1. Quiet standing is attempted afterwards, as a fine-tune.

**Result of attempt 0 (CPU eval, 100 seeds, deterministic policy, no randomization):**
- Median survival 20.5 s; 97 % of episodes alive at 5 s, 73 % at 15 s, 33 % at 30 s.
- Constant shuffling with 4.7 m median drift and 41° median peak tilt.
- Hip-roll RMS torque 0.58 N·m = 1.24 × vendor rated; 85 over-speed events.
- Brax's in-training eval (stochastic actions) showed only ~1.6 s episodes. Training evals now use the
  deterministic policy.

## E6 — Standing, attempt 1: stepping in place (`stand_A8_step_nodr_s0`, task `stand_v1`)
- A 8-DoF, no randomization, 60 M steps (21 min), γ = 0.99.
- CPU eval, 100 seeds:
  - **96 % alive at 30 s** (98 % at 15 s), median tilt 15°, drift 1.5 m.
  - It ignores the 1.6 Hz clock and micro-steps at ~11 steps/s, counting only touchdowns after ≥ 60 ms
    of air; plus ~106 shorter bounces per 30 s.
  - **Not hardware-plausible:**
    - hip-roll RMS torque 0.61 N·m = **1.29 × the vendor rated torque** (0.47 N·m); knee ≈ 1.0 × rated;
    - 85 over-speed events;
    - duty saturation 4 % of the time.
- Video: `assets/videos/stand_A8_step_nodr_s0__stand_seed10000.mp4` (seed 10000, the first eval
  seed). The posture is crouched, knees well bent, constantly stepping.
- B (rocker) was trained with the same v1 recipe for a matched comparison (`stand_B8_step_nodr_s0`).
- Next, v2: add a sustained-torque penalty (quadratic above 1/3 stall, weight −3), a stronger torque
  penalty (−0.2), over-speed (−2) and saturation (−1) penalties, and action rate −0.05.

## E7 — Standing v1, rocker B (`stand_B8_step_nodr_s0`): same recipe and budget as E6
- Training: B learns far faster. Deterministic in-training eval reached 930/1000-step episodes at
  20 M steps; A was at ~2 s at the same point.
- CPU eval, 100 seeds, full-contact model:
  - 94 % alive at 15 s and 85 % at 30 s (A: 98 % / 96 %);
  - 10.5 steps/s; drift 2.5 m;
  - hip-roll RMS torque 0.64 N·m = 1.37 × vendor rated.
- B transfers *worse* to the full-contact, higher-precision CPU model than A does. Neither foot changes
  the balance strategy: both micro-step rapidly. Hypothesis "B makes stationary balance easier":
  **not supported so far** (no stationary standing in either).

## E8 — Standing v2: hardware-plausibility pressure (`stand_A8_v2_nodr_s0`, task `stand`)
- v1 plus: sustained-torque penalty (quadratic above 1/3 stall, −3), torque −0.2, over-speed −2,
  saturation −1, action rate −0.05. A 8-DoF, no randomization, 60 M steps.
- CPU eval, 100 seeds:
  - **99 % alive at 30 s** (1 fall in 100), median peak tilt 7.7°, drift 0.7 m, 7.9 steps/s;
  - duty saturation 0.7 % and 1 over-speed event, against 4.1 % and 85 in v1;
  - RMS torque: hip roll 0.40/0.37 N·m, hip pitch 0.38, knee 0.41/0.40 N·m, i.e.
    **0.79–0.88 × the vendor rated torque on every joint** (0.28–0.29 × stall).
- It passes the vendor-rated sustained check. It fails the deliberately conservative scenario
  (0.25 × stall = 0.35 N·m) by ~15 %. Peak torque touches stall briefly (impacts / recoveries).
- *Correction (logged honestly):* an earlier draft of E5–E8 read the eval's `rms_tau` (N·m) as a
  fraction of stall. The eval now reports `rms_tau` [N·m], `rms_tau_frac` and `rms_over_rated`
  explicitly; `scripts/backfill_rms.py` updated the older files.
- Still micro-stepping (0 % stationary episodes).

**Robustness of `stand_A8_v2_nodr_s0`** (trained *without* randomization; 50 held-out seeds per
condition, `scripts/robustness.py`, plot `results/plots/robustness_stand.png`). Alive at 15 s:

| condition | alive |
|---|---|
| nominal | 1.00 |
| full domain randomization | 1.00 |
| friction 0.5 | 1.00 |
| friction 0.35 | 0.90 |
| mass ×1.2 | 1.00 |
| servo strength ×0.75 | 0.98 |
| latency 12 ms | 0.96 |
| 5° slope | 1.00 |
| ideal sensors | 0.94 |
| sensor noise ×2 | **0.66** |
| pushes σ = 0.3 m/s | 0.84 |
| pushes σ = 0.5 m/s | **0.40** |

The weak spots are sensor noise and larger pushes. Training used nominal noise (×1) and no pushes.

## E9 — Walking v2 (`walk_A8_dr_s0`): balances but does not walk — stopped at 60 M steps
- A 8-DoF, domain randomization, task `walk_v2` (v2 hardware terms; track_lin 1.5 with σ² = 0.01;
  alive 1.0). Commands vx ∈ [0, 0.2], vy ± 0.05, wz ± 0.3; 20 % zero commands.
- Training: episodes reached ~980/1000 steps by 30 M. Mean forward velocity stayed at −0.07 m/s from
  20 M to 60 M, so tracking never started.
- CPU walk test (0.15 m/s command, 20 s, 30 seeds):
  - 93 % alive at 15 s;
  - median displacement −0.18 m;
  - reached 1 m: 10 %; reached 2 m: 0 %.
- Failure taxonomy: **reward exploitation / local optimum**. Stepping in place earns alive + gait-phase
  reward safely; tracking pays at most 1.5 per step with a narrow kernel.
- Stopped by me at 60 M to save GPU time (queue shows FAILED rc = 143; reason in `done.json`).
  Restarted as v3 (track 3.0, σ² = 0.02, alive 0.5), warm-started from this checkpoint.

## E10 — Walking v3 (`walk_A8_v3_dr_s0`, warm start from E9): tracks speed, not heading
- Changes from v2: track 3.0 with σ² = 0.02, alive 0.5. 140 M steps with domain randomization.
- Mean forward velocity in training rose from −0.075 to +0.02 m/s within 20 M steps (averaged over all
  commands).
- CPU walk test (0.15 m/s command, 30 seeds) at 111 M:
  - body-frame vx 0.12 m/s (80 % of command);
  - with no heading control, displacement along the initial heading is ~0;
  - **median heading change 83° in 20 s**.
- Evaluation fix: the walk test now uses a *heading hold* outer loop, the operator/joystick equivalent:
  yaw-rate command = clip(1.0 × heading error, ±0.3). It reports distance along the *initial*
  heading, net displacement and heading change separately. With the hold, at 111 M:
  - ≥ 1 m: 40 %; ≥ 2 m: 7 %;
  - median heading change still 83°: the policy barely follows yaw-rate commands.
- v4 (next): the training env issues the yaw-rate command from a heading error (legged-gym heading
  command, same gain as the eval hold) and raises yaw tracking to 1.5. Fine-tuned from v3 (80 M),
  and trained from scratch for A and B (150 M each) for the matched comparison.

**Final v3 (`walk_A8_v3_dr_s0`, 140 M, CPU, 50 seeds, heading hold):**
- 94 % alive at 15 s;
- ≥ 0.25 m 76 %, ≥ 0.5 m 60 %, **≥ 1 m 42 %, ≥ 2 m 2 %**;
- body vx 0.12 m/s; median heading change 79°; 8.5 steps/s.
- Turn test (vx = 0, wz = +0.3 rad/s for 10 s, ideal +172°): median −22° (range −171…+147°).
  **Yaw commands are ignored.**

## E11 — Walking v4 fine-tune (`walk_A8_v4_ft`, 80 M from v3): regression
- Heading-command env, yaw tracking weight 1.5 (σ² = 0.05).
- The yaw-tracking reward stayed at ~0.09 / 1.5 per step for the whole run, i.e. a yaw-rate error of
  ~0.37 rad/s from the jittery micro-stepping, so the kernel gave almost no gradient.
- CPU, 50 seeds:
  - 100 % alive at 15 s;
  - forward progress collapsed: vx 0.05 m/s, **0 % ≥ 1 m**;
  - turn test median −77°.
- Sign conventions were re-checked end to end (command → IMU yaw rate → heading hold → observation
  index); this was not a sign bug.
- Failure taxonomy: reward design. The fine-tune traded tracking for survival.
- v5 (next, from scratch, A and B matched at 150 M): yaw-rate kernel σ² = 0.25 plus a direct
  heading-hold reward exp(−err²/0.1).

**Other gates for the current best policies (CPU, 50 seeds):**
- `walk_A8_v3_dr_s0` start/stop (fwd 0.15 → stop → fwd, 6 s each, heading hold):
  - **40 % success** (target ≥ 50 %);
  - median vx 0.118 / 0.024 / 0.117 m/s in the three windows; 98 % alive.
- `walk_A8_v3_dr_s0` on the stand test: 86 % alive at 30 s, 8.3 steps/s.
- `stand_A8_v2_nodr_s0` push test (σ = 0.3 m/s base-velocity kicks every 2–4 s, 20 s): 84 % alive at 15 s.

## E12 — Standing v2, rocker B (`stand_B8_v2_nodr_s0`): A and B are indistinguishable
Same recipe and budget as E8 (60 M steps, no randomization). CPU eval, 100 seeds:

| | A (ball) | B (rocker) |
|---|---|---|
| alive @15 s / @30 s | 99 % / 99 % | 100 % / 99 % |
| steps/s | 7.9 | 8.3 |
| stationary episodes | 0 % | 0 % |
| drift (30 s) | 0.71 m | 1.02 m |
| median peak tilt | 7.7° | 8.3° |
| duty saturation | 0.7 % | 1.1 % |
| RMS torque / vendor rated (max joint) | 0.88 | 0.95 |

Conclusion so far: the 32 × 13 mm rocker gives **no measurable standing advantage** over the 16 mm
ball once both are trained with the same recipe. Both balance by micro-stepping at ~8 Hz.

**Robustness, A vs B, standing v2** (same 50 held-out seeds per condition; alive at 15 s):

| condition | A ball | B rocker |
|---|---|---|
| nominal | 1.00 | 1.00 |
| ideal sensors | 0.94 | 1.00 |
| domain randomization | 1.00 | 0.94 |
| friction 0.5 | 1.00 | 1.00 |
| friction 0.35 | 0.90 | 0.98 |
| mass ×1.2 | 1.00 | 1.00 |
| strength ×0.75 | 0.98 | 0.98 |
| latency 12 ms | 0.96 | 1.00 |
| noise ×2 | 0.66 | 0.50 |
| slope 5° | 1.00 | 1.00 |
| push σ 0.3 | 0.84 | 0.74 |
| push σ 0.5 | 0.40 | 0.28 |

No consistent winner. B is a bit better at low friction and latency; A is better under sensor
noise and pushes. Both are weakest under doubled sensor noise and larger pushes.

## E13 — Standing v2 + domain randomization (`stand_A8_v2_dr_s0`, 80 M, warm start from E8)
- Randomized: tip friction 0.5–1.2; link mass ±15 %; pelvis CoM ±5 mm; armature 0.6–1.6×; gear friction
  0.5–2×; servo strength 0.8–1.05; speed 0.8–1.05; kp 0.75–1.25; latency 0–12 ms; backlash-like
  deadzone 0–0.9°.
- Robustness, same seeds as E8 (alive at 15 s, no-DR → DR):

| condition | no DR | DR |
|---|---|---|
| nominal / ideal sensors / DR / friction 0.5 / mass ×1.2 / slope 5° | all ≥ 0.94 | 1.00 |
| friction 0.35 | 0.90 | 0.78 |
| latency 12 ms | 0.96 | 0.86 |
| noise ×2 | 0.66 | 0.48 |
| push σ 0.3 | 0.84 | 0.78 |
| push σ 0.5 | 0.40 | 0.32 |

- **Negative result: randomization fine-tuning did not improve robustness, and was slightly worse on
  the hard conditions.** Likely causes:
  - the DR set did not include the failure modes that matter (sensor-noise scale, pushes);
  - the warm-started fine-tune may have specialized.
- Untested next steps: train with noise ×1.5–2 and pushes from scratch.

## E14 — Walking v5 from scratch (`walk_A8_v5_dr_s0`, A 8-DoF, DR): **first passing walk gate**
- v4 plus a yaw-rate kernel σ² = 0.25 (track_ang 1.0) and a direct heading-hold reward (1.0,
  σ² = 0.1); heading command in training; 150 M steps planned.
- Training: the heading reward reached 0.90 / 1.0 per step by 20–30 M. Episodes became shorter at
  40 M (len 395) while pushing speed.
- CPU eval of the **30 M checkpoint**, 100 seeds. Checkpoint chosen as the first one with full-length
  deterministic training-eval episodes, *not* by CPU eval seeds; the final checkpoint is reported
  separately.

| test | result |
|---|---|
| walk 0.15 m/s, 20 s, heading hold | **≥ 2 m: 86 %**, ≥ 1 m: 98 %, alive @15 s 98 %, vx 0.115 m/s, heading drift 3.9°, 6.0 steps/s |
| stand (zero command, no heading hold), 30 s | **100 % alive @30 s**, 4.8 steps/s, drift 1.1 m |
| start / stop / start | **5 % success**; vx 0.117 / **0.088** / 0.125 m/s, i.e. it never stops |
| turn (wz = +0.3, open loop, 10 s) | median +28° (ideal +172°); follows heading-style commands only |
| actuator load | RMS 0.51–0.90 × vendor rated on every joint; peak at stall briefly; duty saturation 0.7 %; 8 over-speed events in 100 episodes |

- Failure taxonomy for start/stop: **inability to stop**. The tracking kernel barely penalizes
  0.09 m/s, so a zero command still yields walking.
- v6 fine-tune queued: 35 % zero commands plus a sharp zero-command stop reward (2.0, σ² = 0.002).

**E14 addendum: checkpoint selection and late-training regression (read this before quoting E14).**
The final checkpoint (150 M) regressed. The in-training deterministic eval reward peaked at 50 M
(4319) and settled around 3500. From here on the selection rule is declared:
*highest in-training (GPU, separate seeds) deterministic eval reward*
(`scripts/best_checkpoint.py`). CPU, 100 seeds each:

| checkpoint | walk ≥ 2 m | ≥ 1 m | vx @ 0.15 cmd | heading drift | start/stop | stand 30 s | steps/s |
|---|---|---|---|---|---|---|---|
| 30 M (*post-hoc pick*: n = 30 CPU results were seen first) | **86 %** | 98 % | 0.115 | 3.9° | 5 % | 100 % | 6.0 |
| **50 M (declared rule)** | 0 % | 79 % | 0.065 | 10° | **44 %** | 99 % | 5.1 |
| 150 M (final) | 0 % | 4 % | 0.078 | 61° | 33 % | 99 % | 9.2 |

- Honest reading: the ≥ 2 m gate is passed only by a checkpoint chosen after looking at CPU
  results. The rule-selected checkpoint walks more slowly (≥ 1 m in 79 %) but can stop. There is a
  walk-speed versus stopping trade-off, plus training instability after ~50 M (forgetting / reward
  exploitation).
- v6 (fine-tune from 30 M with an explicit stop reward) tests whether both can coexist.

**Walking robustness, `walk_A8_v5_dr_s0` @50 M** (rule-selected; 30 seeds per condition; walk
0.15 m/s with heading hold; share reaching 1 m / fall rate):

| condition | ≥ 1 m | falls |
|---|---|---|
| nominal | 0.73 | 0.03 |
| ideal sensors | 1.00 | 0.00 |
| mass ×1.2 | 1.00 | 0.00 |
| strength ×0.75 | 1.00 | 0.00 |
| slope 5° up | 0.73 | 0.03 |
| full DR | 0.67 | 0.00 |
| noise ×2 | 0.67 | 0.30 |
| latency 12 ms | 0.60 | 0.00 |
| push σ 0.3 | 0.60 | 0.27 |
| friction 0.5 | 0.43 | 0.00 |
| push σ 0.5 | 0.33 | 0.83 |
| **friction 0.35** | **0.03** | 0.10 |

Interpretation:
- Walking progress is limited by tip **traction**. The ball tips slip rather than push on low-friction
  floors. That makes the rubber cap and floor choice first-order hardware risks.
- Weaker servos (×0.75) and +20 % mass do not hurt. Torque is not the limiter in the nominal walk.

## E15 — Walking v5, rocker B (`walk_B8_v5_dr_s0`), matched to E14 (150 M, same recipe and seeds)
The declared rule selected the 30 M checkpoint for B (50 M for A). CPU, 100 seeds:

| | A ball (50 M) | B rocker (30 M) |
|---|---|---|
| walk ≥ 1 m / ≥ 2 m | 79 % / 0 % | 86 % / 6 % |
| vx @ 0.15 cmd | 0.065 | 0.073 |
| heading drift | 10° | 12° |
| start/stop success | 44 % | 38 % |
| stand alive @30 s | 99 % | 98 % |
| steps/s (walking) | 5.1 | 6.8 |

As with standing, **foot shape does not change the outcome** within run-to-run noise. Both show the
same late-training regression. Full table: `results/compare_walk_v5.md`.

## E16 — Walking v6 stop fine-tune (`walk_A8_v6_ft`): collapsed — stopped at ~25 M
- Started from v5 @30 M: 35 % zero commands plus a sharp zero-command stop reward (2.0, σ² = 0.002).
- Episode length went 989 → 26 steps at 10 M → 97 at 20 M. Stopped by me (FAILED rc = 143; reason
  in `done.json`).
- Diagnosis: `train.py` warm-started the *policy* but re-initialized the *critic*
  (`restore_value_fn=False`), at the full learning rate (3e-4), with a changed reward. The first
  updates with a random critic destroyed the policy. The v4 fine-tune regression (E11) likely shares
  this cause.
- Fix: warm starts now restore the critic too, and fine-tunes use lr = 1e-4. Retried as
  `walk_A8_v6b_ft`. The queued quiet-stand fine-tunes also use the fix.

## E17 — Walking v6b (`walk_A8_v6b_ft`): **all primary gates pass** (selected by the declared rule)
- Fine-tuned from v5 @30 M with the critic restored and lr 1e-4; 40 M steps; task `walk_v6` (stop
  reward + 35 % zero commands).
- No collapse; in-training eval reward reached 4751 (the best of any walk run). The rule selected
  40 M (the final checkpoint). CPU, 100 held-out seeds each, full-contact model:

| gate (spec target) | result |
|---|---|
| free stand ≥ 15 s (≥ 70/100) | **100/100**, also 100/100 at 30 s; drift 0.35 m; 4.8 steps/s |
| walk ≥ 2 m (≥ 70/100), 40 s test | **100/100**, 0 falls; median 3.4 m; vx 0.086 m/s at a 0.15 command; heading drift 3.8° |
| walk, 20 s test | ≥ 1 m 99 %, ≥ 2 m 5 % (speed-limited: 2 m needs ~24 s at 0.084 m/s) |
| start → stop → start (≥ 50/100) | **79/100**; median vx 0.091 / **0.012** / 0.089 m/s |
| hardware plausibility | RMS **0.61–0.82 × vendor rated**; peaks reach stall briefly; duty saturated 0.4 %; 0–2 over-speed events per 100 episodes |

- Same 40 s test for the v5 policies: A @50 M ≥ 2 m 85 %; B @30 M 92 %.
- Remaining gaps:
  - commanded speed is not reached (57 % of 0.15 m/s);
  - open-loop turning is not tracked (heading hold needed);
  - never stationary (micro-steps ~5 /s while "standing").

**Walking robustness, `walk_A8_v6b_ft` @40 M** (30 seeds per condition; share reaching 1 m in 20 s
/ fall rate; v5 @50 M from E14 in brackets):

| condition | v6b ≥ 1 m | falls | (v5 ≥ 1 m) |
|---|---|---|---|
| nominal | 1.00 | 0.00 | (0.73) |
| ideal sensors | 1.00 | 0.00 | (1.00) |
| mass ×1.2 | 1.00 | 0.00 | (1.00) |
| strength ×0.75 | 1.00 | 0.00 | (1.00) |
| slope 5° | 1.00 | 0.00 | (0.73) |
| latency 12 ms | 0.93 | 0.00 | (0.60) |
| full DR | 0.90 | 0.00 | (0.67) |
| noise ×2 | 0.80 | 0.20 | (0.67) |
| friction 0.5 | 0.67 | 0.00 | (0.43) |
| push σ 0.3 | 0.67 | 0.40 | (0.60) |
| friction 0.35 | 0.27 | 0.00 | (0.03) |
| push σ 0.5 | 0.27 | 0.80 | (0.33) |

## E18 — Quiet-stand fine-tune A (`stand_still_A8_ft`, 30 M, lr 1e-4, from stand v2)
- Gait-phase shaping off; both feet down +0.5; touchdown penalty −0.5 (plus the v2 hardware terms).
- CPU, 100 seeds:
  - 94 % alive at 30 s;
  - **still 6.7 steps/s; 0 % stationary episodes**;
  - drift 0.87 m.
- With point contacts the policy cannot hold a still stance even when every footfall is penalized.
  This is the spec's anticipated negative result: **true quiet standing is not achieved; balance is
  dynamic micro-stepping.**

- Rocker B, same fine-tune (`stand_still_B8_ft`): 99 % alive at 30 s, **7.6 steps/s, 0 % stationary**.
  The 32 mm rocker does not enable quiet standing either.

## E19 — Actuator ablation: conservative sustained cap (`stand_A8_v2strict_nodr_s0`, rated_frac = 0.25)
- Same as stand v2, but the sustained-torque penalty starts at 0.25 × stall (0.35 N·m) instead of
  0.33. 45 M steps.
- CPU, 100 seeds:
  - 97 % alive at 15 s, **95 % at 30 s**;
  - 5.1 steps/s (v2: 7.9) and far fewer bounces (24 vs 101);
  - drift 0.63 m.
- RMS torque / stall: hip roll 0.247, hip pitch 0.215, knee 0.221, yaw 0.119. **Meets the conservative
  0.25 × stall scenario on every joint** (v2: 0.27–0.29). Peaks still reach stall briefly.
- So standing does **not** depend on optimistic motor ratings; it survives the pessimistic sustained
  cap at a small cost (99 → 95 %).

## E20 — Form-factor ablation: slim Dynamixel hips (`stand_slimA8_v2_nodr_s0`, XC330/XL330, 45 M)
- Geometry: hips 88 mm = **24.0 % of height** (meets the aesthetic target that the Feetech baseline
  misses), thighs 36 mm, 665 g. The yaw case touches the shell (flagged in Gate A). Servos ≈ $715.
- CPU, 100 seeds:
  - 98 % alive at 15 s, **91 % at 30 s**; 3.4 steps/s;
  - RMS torque ≤ 0.29 × stall;
  - **2,046 over-speed events** (Feetech v2: 1), with knee speed reaching **1.63 × no-load** (the joint is
    back-driven faster than the motor can turn); duty saturation 1.7 %.
- Conclusion: the slim, premium servo set stands but is **speed-limited**, exactly as the Gate B audit
  predicted (XC330 knee feasible to ~1.5 steps/s). Narrower hips cost both money and control margin.
  Walking with this set was not trained (out of time).

## E21 — Robustness training from scratch (`stand_A8_v2_robust_s0`, DR + sensor noise ×1.5 + 0.3 m/s pushes, 45 M)
- CPU nominal (100 seeds): 98 % alive at 30 s; 7.5 steps/s; drift 0.6 m.
- Robustness, same 50 seeds as E8/E13 (alive at 15 s, v2-plain → robust):

| condition | v2 plain | robust |
|---|---|---|
| noise ×2 | 0.66 | **0.92** |
| DR | 1.00 | 1.00 |
| strength ×0.75 | 0.98 | 1.00 |
| latency 12 ms | 0.96 | 0.94 |
| friction 0.5 | 1.00 | 0.94 |
| friction 0.35 | 0.90 | **0.46** |
| push σ 0.3 | 0.84 | 0.80 |
| push σ 0.5 | 0.40 | **0.20** |

- Result: **a trade-off, not a fix.** Training at the noise level fixed noise sensitivity.
  Training with pushes did *not* improve push recovery, and low-friction standing got worse.
- Plausible explanation: point feet give very little recovery authority, and a push budget of
  45 M steps was too small. **Push recovery remains the weakest capability.**

## E22 — Seed variability, standing v2 A (`stand_A8_v2_nodr_s1`, seed 1, 45 M vs seed 0 at 60 M)
- CPU, 100 seeds:
  - seed 1: 98 % alive at 15 s / **97 %** at 30 s; 7.8 steps/s; drift 0.86 m; 0 over-speed events;
  - seed 0: 99 % / 99 %; 7.9 steps/s; drift 0.72 m.
- The standing result reproduces across training seeds.
- Walking was not repeated across seeds (out of GPU time). Its variability is the main untested
  uncertainty.

## GPU-time accounting (00:47 – 08:19, one job at a time)
21 training runs, about 7.5 h of GPU time:
- 3 failed or stopped: E9 stopped, E16 collapsed, E11 regressed;
- 18 completed and evaluated.
- The queue runner and watchdog never ran two jobs at once.
- No crashes or stalls needed the watchdog. Two runs were stopped by me with reasons recorded.

## Lessons (process, not just results)
1. **Audit the actuators before training.** The static and swing torque-speed audit (E2) rejected
   the 12 V servo before any GPU time was spent, and the slim-Dynamixel run (E20) later confirmed
   its speed prediction. A cheap first-principles check saved a dead-end training campaign.
2. **Don't fight the physics with the reward.** Rewarding quiet standing on point feet (E5) taught
   the policy to fall slowly. Allowing stepping (E6) solved standing immediately.
3. **Put hardware limits in the reward, not only in the simulator.** Clipping torque in the
   simulator alone still produced policies that ran 1.3 × over rated torque (E6). The
   sustained-torque penalty fixed it, and survival improved too (E8).
4. **Make the evaluation match the question.** Brax's stochastic in-training evals understated
   stand time (1.6 s vs 20 s), so they were switched to deterministic. Walking without a heading
   hold measured "distance along the initial heading" for a policy that drifted 80°. The
   heading-hold outer loop and the separate drift metric fixed that (E10).
5. **Fine-tune with the critic restored and a lower learning rate.** A random critic plus a changed
   reward destroyed a good policy within 10 M steps (E16), and probably caused E11 as well.
6. **Declare checkpoint selection before looking at test results.** Walk policies peaked early and
   regressed (E14). Picking the best checkpoint after seeing test results inflated one result
   (86 % vs 0 % under the rule).
7. **Randomization isn't free robustness.** Domain randomization that does not include the actual
   failure modes (noise scale, pushes) did nothing (E13). Training on noise fixed noise, but pushes
   on point feet need more than 45 M steps or a different strategy (E21).
8. **Run one GPU job at a time, with a watchdog.** A queue runner with a 20-minute stall check
   kept the GPU busy for 7.5 h with no crashes.

## Artifacts kept in the repository
- **Every run:** `runs/*/config.json` (exact arguments, reward weights and environment config) and
  `progress.jsonl` (in-training evaluation curve).
- **Weights:** only `runs/walk_A8_v6b_ft/params.pkl` (the selected policy; identical to its 40 M
  checkpoint) and `runs/walk_A8_v5_dr_s0/params_0030M.pkl` (its fine-tune starting point). The other
  checkpoints and the mp4 videos referenced above are not committed. Regenerate them with
  `scripts/train.py` and `scripts/render_rollout.py`; the run configs hold the exact settings.
- **Evaluation:** every evaluation JSON quoted in this log is in `results/eval/`.

