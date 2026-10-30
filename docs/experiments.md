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


---

# Round 2 (2026-10-23, afternoon): feet control experiment, then ball-tip improvements

Question from the user: is footlessness the real reason the robot never stands still and walks with tiny
steps, or is something else (reward, servos, control) responsible? Method:
1. **Control experiment.** Give the same frame ankles and flat feet, and train it with *the same recipe*.
2. **Ball-tip improvements.** Apply what the control teaches, without abandoning the ball-tip form factor.

The ball-tip baseline A model is untouched: `models/baseline_A_8dof.xml` is byte-identical before and
after this round (checked by md5). New frames are separate models.

**New models** (`configs/geometry.yaml`, `scripts/regen_train_models.py`):

| model | what | DoF | mass |
|---|---|---|---|
| `footed_F40_12dof` / `F60` / `F90` | baseline A legs + ankle pitch & roll + flat rubber sole 40×20 / 60×32 / 90×45 mm (≈ 11 / 16 / 24 % of height) | 12 | 1.09–1.12 kg |
| `footed_F60_8dof` | rigid 60×32 mm sole, no ankle: is it the foot or the ankle? | 8 | 0.88 kg |
| `baseline_P20_8dof` / `P30` | **ball-tip form factor with a flat rubber pad instead of the ball** (20×14 / 30×18 mm), rigid, no ankle (user suggestion) | 8 | 0.86 kg |

Confound, stated up front: the footed control frame uses STS3215-C046 servos at the ankles (so ankle
torque cannot be the limiting factor), which adds ~250 g low on the legs. Packaging is not enforced for
the control frame; it is a test rig, not a design candidate.

**New metrics** (`scripts/evaluate.py`): after a 2 s settle, `liftoffs` (any tip or sole leaving the floor,
however briefly), `tip_travel_m` (horizontal sliding/stepping of both tips), `base_speed_rms`.
`quiet_frac` = survived **and** zero lift-offs **and** < 2 cm tip travel. This is stricter than round 1's
`stationary_frac`, which ignored airborne blips shorter than 60 ms. Walking adds `step_length_m` =
forward distance / steps.

## E23 — Passive pose hold (no policy): what the contact geometry gives for free
Nominal joint targets held by the servo model, 10 s, 5 seeds; "clean" = no reset noise, "noisy" = round-1
reset noise (joints σ 0.08 rad, tilt σ 0.04 rad). Survival in seconds:

| model | clean start | noisy start |
|---|---|---|
| ball A (round-1 baseline) | 0.76 ×5 | 0.4–0.7 |
| pad P20 (20×14 mm) | **10 ×5** | 0.5–1.2 |
| pad P30 (30×18 mm) | **10 ×5** | 1 of 5 survives |
| rigid sole F60 (no ankle) | 10 ×5 | 4 of 5 survive |
| footed F40 / F60 / F90 (ankles) | 10 ×5 | 2 / 4 / 4 of 5 survive |

A ball tip has no static equilibrium: two point contacts leave one unactuated degree of freedom, a
rotation about the line between the tips. Any flat contact, even a 20 mm pad, creates one. The pad's
margin is tiny, though (noise knocks it over), so it still needs active balance. The difference is that
the balance can now be quiet.

## E24 — Control: footed F60 (12 DoF), quiet-stand recipe from scratch (`ctl_stand_F60_12`, task `stand_still`, 45 M)
- Recipe identical to the ball run E25 (round 1's `stand_still`: both feet +0.5, touchdown −0.5, no gait
  shaping, hardware penalties, termination −5, alive +1).
- **Checkpoint-rule amendment** (made before any CPU result of this run was seen): the rule picked the
  *untrained* step-0 checkpoint, because a near-zero-action policy on flat feet holds pose passively and
  scores highest. `scripts/best_checkpoint.py` now ignores step 0. Selected: 45 M. Both are reported.

| checkpoint | alive @30 s | no-lift stance | strict quiet | lift-offs | steps/s | tip travel (28 s) | heading drift |
|---|---|---|---|---|---|---|---|
| 45 M (rule) | 96 % | **95 %** | 0 % | **0** | 0.03 | 0.45 m | 107° |
| 0 M (untrained) | 82 % | 82 % | 77 % | 0 | 0 | 0.01 m | small |

- Diagnosis of the 0 % strict quiet (seed 10000): neither sole ever leaves the floor, and joint motion
  stays < 1° std. The robot slowly **pirouettes** (~0.06 rad/s) by twisting its soles, driven at the
  1.6 Hz gait-clock frequency. That is a reward gap: zero-command yaw was barely penalized (kernel
  σ² = 0.05). It is not a balance problem. The phase-2 quiet task narrows the yaw kernel to 0.005.
- **Result: with feet, the policy stops stepping entirely.**

## E25 — Control: ball A (8 DoF), the identical recipe from scratch (`ctl_stand_A8`, 45 M, rule → 33 M)

| | footed F60 (E24) | **ball A** |
|---|---|---|
| alive @30 s | 96 % | 97 % |
| no-lift stance | 95 % | **0 %** |
| lift-offs per episode (median) | 0 | **221** |
| steps/s | 0.03 | **6.2** |
| tip travel in 28 s | 0.45 m | **12.6 m** |
| base speed RMS | 0.03 m/s | 0.18 m/s |
| RMS torque / rated (max joint) | 0.53 | 0.73 |

**Same reward, same budget, same legs: the ball tip micro-steps, the flat foot does not.** Round 1's
constant stepping is caused by the point contact, not by the reward design (which in round 1 did
explicitly shape stepping) and not by the servos.

## E26 — Control: foot size (`ctl_stand_F40_12`, `ctl_stand_F90_12`; same recipe, 45 M each)
100 held-out seeds; rule-selected checkpoints (33 M for both).

| foot | alive @30 s | no-lift stance | strict quiet | tip travel (28 s) | base speed RMS | heading drift |
|---|---|---|---|---|---|---|
| ball A (E25) | 97 % | 0 % | 0 % | 12.6 m | 0.18 m/s | 94° |
| F40 40×20 mm (11 % of height) | 98 % | 94 % | 0 % | 0.68 m | 0.041 m/s | 27° |
| F60 60×32 mm (16 %) (E24) | 96 % | 95 % | 0 % | 0.45 m | 0.032 m/s | 107° |
| F90 90×45 mm (24 %) | 92 % | 90 % | **12 %** | **0.03 m** | **0.009 m/s** | 27° |

- **Any foot stops the stepping.** Even the 40 mm sole gives a no-lift stance in 94 % of episodes.
- **Bigger feet make the stance quieter** (sliding drops by ~20× from F40 to F90). Bigger feet are the only
  size to reach strict quiet in some episodes.
- All 8 F90 falls happen in the first 2.4 s (tilt ~42°): the randomized start leaves a big sole on its
  edge. Every F90 episode that survived 2.4 s stood the full 30 s.

## E27 — Control: is it the foot or the ankle? Rigid F60 sole, no ankle (`ctl_stand_F60_8rigid`, 8 DoF, 45 M)
- Rule-selected 45 M: **95 % alive @30 s, 94 % no-lift stance**, 0 median lift-offs, tip travel 0.55 m,
  base speed 0.031 m/s, heading drift 12°.
- Same as the 12-DoF F60 with ankles (96 % / 95 %). **The flat contact, not the ankle, removes the stepping.**
  The hip and knee servos can tilt a rigid sole well enough to stand. This motivates the flat-pad leg
  (`baseline_P20/P30`): a rigid flat contact inside the ball-tip silhouette.

## E28 — Control: footed F60 walking with the ball's walk recipe (`ctl_walk_F60_12`, task `walk` = v5, DR, 100 M)
- The rule picked 91 M. (The untrained step 0 scored highest, 4877: on flat feet, standing still with
  good heading earns most of the walk reward. The amended rule ignores step 0.)
- CPU, 100 seeds:

| | footed F60 @91 M | ball v5 @50 M (same recipe) | ball v6b (round-1 best) |
|---|---|---|---|
| walk ≥ 2 m in 40 s | 83 % | 85 % | 100 % |
| mean vx at a 0.15 command | 0.054 m/s | 0.065 m/s | 0.086 m/s |
| lift-offs in 38 s (median) | 92 | 444 | 462 |
| start/stop success | 43 % | 44 % (E14) | 79 % |
| standing: no-lift stance | 0 % (59 lift-offs) | — | — |

- **Gait anatomy** (`scripts/gait_anatomy.py`, 20 s, 3 seeds): where does foot travel happen?

| | foot travel airborne | foot travel sliding on the floor | lift-offs | mean air time | airborne step |
|---|---|---|---|---|---|
| footed F60 | 0.36 m | **1.59 m** | 28 | 31 ms | 0.4 cm |
| ball v6b | 8.0 m | 1.7 m | 234 | 55 ms | 1.8 cm |

- **The footed robot does not walk; it skates.** It slides its soles forward and lifts them only for
  ~30 ms blips. The ball robot at least swings its tips through the air, in 1.8 cm steps.
- Lesson: **feet fix standing, not walking.** The tiny-step gait is a reward problem as much as a
  contact problem. The v5 reward barely penalizes sliding (slip −4 × v²: ≈ 0.03 per step at 0.08 m/s)
  and weakly enforces alternating contact (0.5). A robot with feet exploits that even more readily,
  since sliding a flat sole is cheap and stable.
- Next: the `walk_stride` recipe (slip −30, stride reward, air-time 0.25 s, 2.4 steps/s clock, contact
  pattern 1.0) on ball A, footed F60 (control) and pad P20.

## E29 — Ball A, quiet stand v2 (`q2_stand_A8`, task `stand_quiet2`, 45 M from scratch)
- `stand_still` plus a lift-off penalty (−1 per lift-off, so short blips count too), a zero-velocity
  stop reward (+1) and a sharp zero-yaw kernel (σ² 0.005, weight 1).
- Rule → 45 M. CPU, 100 seeds: 97 % alive @30 s, **still 210 lift-offs per episode, 6.1 steps/s, 0 %
  no-lift stance**, tip travel 14 m, heading drift 80° (from 94°).
- The third reward design for ball stillness that fails, after E18 (fine-tune from stepping) and E25
  (from scratch). With the hardware-like control loop (50 Hz, 4 ms latency, IMU noise), the policy
  always prefers stepping to falling.
- Queued as an upper bound: `q2_stand_A8_ideal100` (100 Hz control, zero latency, ideal sensors, no
  deadzone). If even that cannot stand still, quiet ball stance is impractical rather than just untrained.

## E30 — Flat pad P20 (ball-tip form factor, 20×14 mm pad), quiet stand v2 (`q2_stand_P20`, 45 M)
- Rule → 22 M. CPU, 100 seeds: 95 % alive @30 s, **210 lift-offs, 5.3 steps/s, 0 % no-lift stance**.
  The final 45 M checkpoint is the same (100 % alive, 278 lift-offs, 5.5 steps/s; 50 seeds).
- So a 20 mm pad by itself did not remove the stepping, unlike the 60 mm rigid sole (E27).
- One visible difference: the untrained policy. On the 60 mm sole it already stood, scoring 3586 at
  step 0, so training began from a quiet stance. On the pad it falls in ~1 s from the randomized start
  (joint noise σ 0.08 rad, tilt σ 0.04), because the pad's passive margin is tiny (E23). PPO then finds
  stepping before it ever experiences quiet balance.
- Hypothesis: **exploration, not physics.** Test: `q2_stand_P20_gentle`, the same recipe with gentle
  starts (σ 0.02 rad / 0.01). It is evaluated under both gentle and standard starts.

## E31 — Flat pad P30 (ball-tip form factor, 30×18 mm pad), quiet stand v2 (`q2_stand_P30`, 45 M): **stands without stepping**
- Training peaked at 22 M (eval reward 4409, 958/1000 steps) and **collapsed** by 45 M (336 steps). The
  declared rule picked 22 M.
- CPU, 100 held-out seeds, standard (noisy) starts:

| | ball A (E29, same recipe) | pad P20 (E30) | **pad P30** | rigid sole F60 (E27, different recipe) |
|---|---|---|---|---|
| alive @30 s | 97 % | 95 % | 94 % | 95 % |
| no-lift stance | 0 % | 0 % | **85 %** | 94 % |
| lift-offs (median) | 210 | 210 | **0** | 0 |
| steps/s | 6.1 | 5.3 | **0.03** | 0.07 |
| base speed RMS | 0.19 m/s | 0.13 m/s | **0.027 m/s** | 0.031 m/s |
| heading drift | 80° | 93° | **17°** | 12° |

- Gait anatomy (30 s, 3 seeds): about 1 lift-off per episode; **body displacement 2 cm**; the pads slide
  0.5 m back and forth (small fidgeting corrections, not stepping). Strict quiet (< 2 cm tip travel) is
  still 0 %.
- The policy holds a **split stance**, one pad ahead of the other (`assets/gifs/pad30_stand.gif`).
  Two flat pads staggered fore-aft form a real support polygon (≈ the stagger × the pad width).
- **Result: a 30×18 mm flat pad at the shin tip, with no ankle or foot, inside the ball-tip silhouette,
  lets the robot stand without stepping.** The 20 mm pad did not, at least not from noisy starts (E30).

## E32 — Pad P20 with gentle starts (`q2_stand_P20_gentle`, reset noise σ 0.02 rad / tilt 0.01, 45 M)
- Rule → 33 M. CPU, 100 seeds, under both start conditions:

| start | alive @30 s | no-lift stance | lift-offs | steps/s | base speed |
|---|---|---|---|---|---|
| gentle (as trained) | 100 % | 0 % | 130 | 1.6 | 0.097 m/s |
| standard (σ 0.08 / 0.04) | 98 % | 0 % | 130 | 1.7 | 0.097 m/s |
| (E30, standard training) | 95 % | 0 % | 210 | 5.3 | 0.13 m/s |

- Partial support for the exploration hypothesis: gentle starts cut stepping by 3× and the result holds
  under standard starts. But **the 20 mm pad still never holds a no-lift stance**; 30 mm does (E31).
- **Pad size threshold:** between 20×14 mm (no) and 30×18 mm (yes) for this 0.86 kg, 38 cm robot with these
  servos. Scale reference: the shin is 22 mm wide at the top and tapers to 11 mm at the tip.

## E33 — Ball A, stride recipe from scratch (`stride_A8`, task `walk_stride`, DR, 100 M): **failed**
- Recipe: 2.4 steps/s clock, air-time target 0.25 s, stride reward (+2 up to 6 cm per step), slip −30,
  contact pattern 1.0, commands 0.05–0.3 m/s. No warm start.
- Training never got past ~3 s episodes (peak 186/1000 steps at 81 M).
- CPU (rule → 81 M, 50 seeds, walk40): 0 % alive @30 s; it lunges forward at 0.38 m/s and falls; 60 %
  reach 1 m. Steps are still only 2.6 cm. Hip roll, hip pitch and knee run at ~1.0 × rated torque.
- Reading: demanding long, airborne, non-sliding steps *from scratch* gives the ball robot nothing to
  hold on to. Every attempt to swing a leg far ends in a fall before the stride reward pays. Next is a
  curriculum: fine-tune the working v6b walker toward longer steps (`walk_stride_ft`, E34).
- The pad stride run was changed for the same reason: v6b transferred onto the P30 model (same
  joints and observations), then fine-tuned with `walk_stride_ft` (E36).

## E34 — Ball A, gentle stride fine-tune from v6b (`stride_A8_ft`, task `walk_stride_ft`, 40 M, critic restored, lr 1e-4)
- Recipe: 2.8 steps/s clock, air-time target 0.2 s, stride reward +1.5 (up to 5 cm), slip −10, commands
  0.05–0.25 m/s, v6 stop reward. Rule → 10 M (in-training eval 4969).
- CPU, 100 seeds each:

| | v6b (round-1 best, re-evaluated) | **stride_A8_ft @10 M** |
|---|---|---|
| walk ≥ 2 m in 40 s | 100 % | 100 % |
| mean vx @ 0.15 command | 0.086 m/s | **0.122 m/s** (81 % of command) |
| start → stop → start | 79 % | **89 %** |
| stand 30 s | 100 % | 100 % |
| steps/s (walking) | 5.7 | 7.0 |
| step length (forward distance / steps) | 1.5 cm | 1.8 cm |
| airborne step / sliding per 20 s (anatomy) | 1.8 cm / 1.7 m | 2.1 cm / 1.0 m |
| RMS torque / rated (max) | 0.82 | 0.82 |

- **A better walker on every round-1 gate, but not a longer stride.** The extra speed comes from cadence
  (7 steps/s), not step length.
- Together with E33 this suggests a ball-tip limit: single support on a point contact is an
  unstabilizable inverted pendulum, so each swing must end within roughly one fall time constant
  (√(l/g) ≈ 0.17 s here). Long swings mean falling. Whether a flat contact relaxes this is tested by
  E35 (footed) and E36 (pad).

## E35 — Control: footed F60 with the stride recipe from scratch (`stride_F60_12`, task `walk_stride`, DR, 80 M)
- Unlike the ball (E33), the footed robot survives the from-scratch stride recipe. Rule → 30 M.
- CPU, 100 seeds: walk40 93 % alive, but **3 % reach 1 m**; vx 0.054 m/s; heading drift 65°; 5.3 steps/s.
  Start/stop 16 %. Standing: 270 lift-offs (it marches in place even at zero command).
- Gait anatomy: real lift-offs (122 ms mean air time, vs 31 ms when skating in E28), but **1.4 cm airborne
  steps**. It marches in place.

| from-scratch stride recipe | ball A (E33) | footed F60 |
|---|---|---|
| survives | no (~3 s) | yes (93 % @30 s) |
| reaches 1 m in 40 s | 60 % | 3 % |
| airborne step | 2.6 cm | 1.4 cm |

- **Conclusion on strides: feet do not produce long strides either.** With the v5 reward they skate (E28);
  with the stride reward they march in place. Short steps are not specific to the ball tip; they come
  from what RL finds easiest under these rewards, servos and this scale. Only the *survivability* of
  long-step training differs (feet survive, the ball falls).
- Remaining idea: step length = speed / cadence. E37 (`cadence_A8_ft`) enforces a 2.4 steps/s clock hard
  while commanding 0.15–0.3 m/s.

## E36 — Pad P30 walking: v6b ball walker transferred onto the pad, then the gentle stride fine-tune (`stride_P30_ft`, 40 M)
- The P30 model has the same 8 joints and observation layout as ball A, so the v6b policy (and critic)
  load directly. Task `walk_stride_ft` (same as E34), lr 1e-4, DR. Rule → 20 M.
- CPU, 100 seeds. E34 (ball, same recipe and initialization) for comparison:

| | ball A (E34) | **pad P30** |
|---|---|---|
| walk ≥ 2 m in 40 s | 100 % | 100 % |
| mean vx @ 0.15 command | 0.122 m/s | **0.157 m/s** (tracks the command) |
| steps/s | 7.0 | **4.25** |
| step length (forward distance / steps) | 1.8 cm | **3.6 cm** |
| airborne step / sliding per 20 s | 2.1 cm / 1.0 m | 3.1 cm / 2.0 m |
| start → stop → start | **89 %** | 44 % |
| standing (this walking policy) | 100 % alive, 363 lift-offs | 99 % alive, 239 lift-offs, 2.2 steps/s |
| RMS torque / rated (max) | 0.82 | 0.79 |

- **The flat pad doubles step length and halves cadence** at the same reward. It also tracks the full commanded
  speed. Stopping on command is worse (44 %). This walking policy keeps stepping at zero command
  (`phase_when_still` on, inherited from v6b), so the pad's quiet stance comes from the separate standing
  policy (E31). Merging the two behaviours into one policy is untested.
- Reading, with E33–E35: the flat contact gives each stance phase some CoP authority, so single support
  lasts longer and steps get longer. The ball cannot. But no configuration produced human-like strides
  (step ≈ 25 % of leg length or more, i.e. ≥ 7 cm) within this budget.

## E37 — Ball A, hard cadence clock (`cadence_A8_ft`, task `walk_cadence`, 30 M from E34 @10 M, lr 1e-4): **new best ball policy, but no longer steps**
- Recipe: contact pattern 3.0 on a 2.4 steps/s clock, commands 0.15–0.3 m/s, stride target 8 cm. The idea:
  step length = speed / cadence, so a held cadence forces long steps.
- Training peaked at 20 M (eval reward 7016) and collapsed at 30 M (411 steps). Rule → 20 M.
- **The policy ignored the clock**: still 7.2 steps/s, 1.8 cm steps (airborne 2.6 cm). The contact-pattern
  reward was traded away for speed tracking and survival.
- But it is the **best ball-tip policy so far on every round-1 gate** (CPU, 100 seeds):

| gate | v6b (round 1) | E34 stride_ft | **E37 cadence_ft @20 M** |
|---|---|---|---|
| stand 30 s | 100 % | 100 % | **100 %** |
| walk ≥ 2 m in 40 s | 100 % | 100 % | **100 %** |
| vx @ 0.15 command | 0.086 | 0.122 | **0.134 m/s** (89 %) |
| heading drift (40 s) | 3.8° | 3.2° | **1.5°** |
| start → stop → start | 79 % | 89 % | **98 %** |
| RMS torque / rated (max joint) | 0.82 | 0.82 | **0.73** |
| peak joint speed / no-load | 0.95–1.0 (knee) | — | **0.79** |
| duty saturation | 0.4 % | — | **0.03 %** |

- Lesson: progressive fine-tuning (v5 → v6b → stride_ft → cadence_ft) kept improving tracking, stopping and
  hardware margins. The step-length goal was not met by any reward lever we tried on the ball tip.

## E38 — Upper bound: ball A quiet stand under ideal control (`q2_stand_A8_ideal100`, 100 Hz, zero latency, ideal sensors, no deadzone, 60 M)
- Same `stand_quiet2` reward as E29. Rule → 50 M. CPU, 100 seeds:

| eval condition | alive @30 s | no-lift stance | lift-offs | drift (30 s) |
|---|---|---|---|---|
| ideal (as trained) | 92 % | **0 %** | 475 | 5.1 m |
| realistic (4 ms latency, nominal noise, deadzone) | 78 % | 0 % | 404 | 1.5 m |

- **Even with perfect sensing and 100 Hz zero-latency control, the ball tip does not learn a quiet stance.**
  It steps (more lift-offs per second, at the higher control rate) and wanders further. Together with
  E18, E25 and E29 (four reward and control variants), ball-tip quiet standing is not an artefact of our
  servo model or control rate. With two point contacts the unactuated tip-over mode makes stepping the
  only robust strategy that RL finds. In principle an acrobot-style feedback balance exists, but it was
  never found.

## E39 — Reproducibility: pad P30 quiet stand, seed 1 (`q2_stand_P30_s1`, 45 M)
- Training only escaped falling at the very end (eval length 60–106 steps from 11 to 34 M, then 991 at 45 M).
  Seed 0 got there by 22 M.
- Rule → 45 M. CPU, 100 seeds: 96 % alive @30 s, but **0 % no-lift stance**: 152 lift-offs, 2.1 steps/s,
  heading drift 87°.
- **The quiet pad stance did not reproduce on seed 1 within the same budget.** The pad *permits* a
  no-lift stance (seed 0: 85 %), but training does not reliably find it. Continued for 30 M from
  45 M (`q2_stand_P30_s1_cont`) to separate "needs more training" from "seed-dependent outcome".
- **Continuation** (`q2_stand_P30_s1_cont`, +30 M, critic restored, rule → 30 M): 100 % alive, lift-offs
  152 → 98, steps/s 2.1 → 0.6, but still **0 % no-lift stance**. The trend is toward quiet, but it is not
  there after 75 M total. **The P30 quiet stance is seed-dependent (1 of 2 seeds) or at best slow to find.**

## Round 2 summary

**Run index**

| # | run | what | outcome |
|---|---|---|---|
| E23 | — | passive pose hold, all contacts | any flat contact has a static equilibrium; the ball has none |
| E24 | `ctl_stand_F60_12` | footed 60 mm, quiet stand | no stepping (95 % no-lift); slow yaw creep |
| E25 | `ctl_stand_A8` | ball, same recipe | 0 % no-lift, 6.2 steps/s |
| E26 | `ctl_stand_F40_12`, `ctl_stand_F90_12` | foot size | all sizes stop stepping; bigger is quieter |
| E27 | `ctl_stand_F60_8rigid` | rigid sole, no ankle | same as with ankle: the contact matters, not the ankle |
| E28 | `ctl_walk_F60_12` | footed walking (ball recipe) | skates its soles; no stepping |
| E29 | `q2_stand_A8` | ball, lift-off penalty | still 210 lift-offs |
| E30 | `q2_stand_P20` | 20 mm pad, quiet stand | still steps |
| E31 | `q2_stand_P30` | 30 mm pad, quiet stand | **85 % no-lift stance** (seed 0) |
| E32 | `q2_stand_P20_gentle` | 20 mm pad, gentle starts | stepping cut 3×, still not quiet |
| E33 | `stride_A8` | ball, stride reward from scratch | falls (~3 s) |
| E34 | `stride_A8_ft` | ball, gentle stride fine-tune | better walker, same 2 cm steps |
| E35 | `stride_F60_12` | footed, stride reward from scratch | marches in place |
| E36 | `stride_P30_ft` | pad, v6b transferred + stride fine-tune | **3.6 cm steps (2× ball), 0.157 m/s** |
| E37 | `cadence_A8_ft` | ball, hard cadence clock | **best ball policy** (98 % start/stop); still 2 cm steps |
| E38 | `q2_stand_A8_ideal100` | ball, ideal 100 Hz control | still 0 % no-lift |
| E39 | `q2_stand_P30_s1(_cont)` | pad P30, seed 1 | quiet stance not reproduced (0.6 steps/s at 75 M) |

**Answers to the round-2 questions**
1. *Does footlessness cause the constant stepping?* **Yes.** Same reward, same legs: every flat foot
   (40–90 mm, with or without an ankle) stands with no lift-offs; the ball never does, even with ideal
   sensing and control. Two point contacts leave an unactuated tip-over mode.
2. *Does footlessness cause the tiny steps?* **Not by itself.** Feet with the same rewards skated
   (E28) or marched in place (E35). A flat pad did lengthen steps 2× at an identical reward and
   initialization (E36), so contact shape contributes, but reward design and training method dominate.
3. *Can the ball-tip form factor stand still?* **The 16 mm ball cannot** in anything we tried. A
   **30 × 18 mm flat pad** at the shin tip, the same silhouette, **can** (E31, 85 % no-lift), but training
   found it on 1 of 2 seeds. A 20 mm pad is below the threshold.
4. *Best ball-tip policy:* `cadence_A8_ft` @20 M (E37). It passes every round-1 gate with more margin.

**Lessons (round 2)**
1. **Use a matched control to separate morphology from reward.** Same recipe and budget on feet and
   ball turned "the robot steps constantly" into a clear statement: point contacts cause it.
2. **Passive tests are cheap and predictive.** The no-policy pose hold (E23) predicted which contacts
   could stand still before any GPU time was spent.
3. **Measure behaviour, not reward.**
   - The footed walker "passed" a 2 m gate by skating (E28).
   - The untrained flat-foot policy topped the reward curve (E24).
   - Gait anatomy (airborne vs sliding travel) and a strict lift-off count were essential.
4. **What the untrained policy can do shapes what PPO finds.** Contacts that stand passively (60 mm
   soles) started quiet and stayed quiet. Marginal contacts (20/30 mm pads) fall first and discover
   stepping. Gentle starts help (E32).
5. **Curricula beat from-scratch rewrites for gait changes.** Stride rewards from scratch failed or
   degenerated (E33, E35). Fine-tuning a working walker improved it step by step (E34, E37) and
   transferred across tip shapes (E36).
6. **Every long run regressed after its peak** (P30 22 → 45 M, cadence 20 → 30 M). The declared
   checkpoint rule, amended once and openly to skip step 0, kept selection honest.
7. **Seeds matter.** The headline pad result reproduced on only 1 of 2 seeds; report that, don't hide it.

**GPU-time accounting (11:03 – 18:26, one job at a time):** 18 runs, about 7.4 h. All 18 completed (none
stalled or crashed); 3 were conclusive failures by design (E33 falls, E35 marches, E38 still steps). The
health watchdog (`scripts/health_watch.sh`) checked the queue runner, GPU utilization and log freshness
every 30 s and reported every 30 min. No intervention was needed.

# Round 3 — the as-built hardware (2026-10-29)

Question: with the leg as actually designed and printed (P0 CAD) and the servos actually bought, can the
two-leg robot stand and walk, and what does it draw from the supply? If not: what to change, from cheap to
expensive (training/settings → parts → stronger servos)?

## E40 — CAD-derived hardware model (`hardware/leg_p0/cad/sim_model.py` → `models/hw_p0_8dof`)
- Every link comes from the printable CAD, not the parametric shells: P08 two-leg pelvis (80 mm hips),
  2 × P0 leg (P03–P07), P30 pad. Mass/COM/inertia per link from the part solids with a print model
  (perimeter walls solid + infill), servo cases at their real poses, horns, screws and cables. Joint axes,
  link lengths and the pad sole come from the CAD kinematics. Joint limits = CAD collision-free range
  (yaw −20…30, roll −14…30, pitch −30…85, knee 0…110). Same joint/geom names as the parametric models, so
  the MJX env, the CPU evaluation and every script run unchanged; visual meshes are the CAD parts.
- Servos as built: **STS3215-C001 (1:345)** on yaw/roll/pitch (1.91 N·m, 5.45 rad/s @7.4 V) and
  **STS3215-C044 (1:191)** on the knee. The C044's listed 27.4 kgf·cm is implausible (more than the 1:345 at
  the same voltage); the model uses 1.5 N·m (current-scaled), 9.0 rad/s. Specs and sources in `configs/actuators.yaml`.
- **907 g** (round-2 sim: 861 g), pelvis 25.6 cm high at stance. Statics: stance needs ≤ 23 % of knee stall
  on one leg (same as the round-2 sim). Passive pose hold with zero action: as good as the round-2 P30 model.
- **The real difference is the hip servos.** The round-2 pad walker's peak joint speeds were 1.1–1.6× the
  C001's no-load speed. And the 1:345 gearbox reflects ~5× more rotor inertia (armature ≈ J_rotor·N², an
  estimate) — effective hip-pitch inertia 60.9 vs 29.2 ×10⁻⁴ kg·m² (2.1×). The direct-drive knee adds 1.3× leg inertia.

## E41 — Standing on the as-built hardware (`hw_stand_ft`, `hw_stand_scratch`, `hw_stand_push`, `hw_stand_push2`)
- `hw_stand_ft`: round-2 P30 quiet stand (E31) transferred, `stand_quiet2`, 40 M → **99 % alive, 90 % no-lift
  stance** (round 2 on its own model: 85 %). From scratch (`hw_stand_scratch`, 45 M): 94 % / 94 % — unlike
  round 2 (1 of 2 seeds), the quiet stance is found from scratch.
- Pushes (0.3 m/s kicks every 2–4 s) knocked both over in ~4 s. Push fine-tune (`--push 0.3 --dr`, 30 M + 20 M):
  **`hw_stand_push2` @20 M — 98 % alive, 98 % no-lift, heading creep 11°, and the full 20 s under pushes
  (73 % survive ≥ 15 s).** Recovery steps drive the hips to their speed limit (overspeed events 777).
- Knee-spec sensitivity (policy replayed with a 1.06 N·m knee, `hw_p0kl`): stand 95 % / 89 % — performance
  holds, but knee RMS torque rises to 0.7–1.0 × rated (heat).

## E42 — Walking, warm-started from the round-2 pad walker (`hw_walk_ft`, `_s1`, `hw_walk_cad`)
- `stride_P30_ft` @20 M transferred, `walk_stride_ft`, DR, lr 1e-4. All fine-tunes peaked at 10–12 M.
- Seed 0: walks (98 % reach 2 m) but **0.076 m/s at a 0.15 command**, 1.5 cm steps at 5.1 steps/s, start/stop
  26 %, 168 overspeed events. Seed 1: 0.116 m/s at 7.4 steps/s. Cadence fine-tune (E37 recipe): 0.088.
- Knee-spec sensitivity: still 94 % reach 2 m with the 1.06 N·m knee; knee RMS torque ≈ 1.0 × rated.

## E43 — Escalation ladder, warm-start recipe (single seeds unless noted)

| change | tier | model | vx @ 0.15 | 2 m | start/stop |
|---|---|---|---|---|---|
| as built (5 runs: seeds 0/1, cadence ft, 60/70 mm hips) | — | hw_p0 (+s60/s70) | **0.088 ± 0.018** | 92–100 % | 25–54 % |
| fully charged 2S (8.4 V: +13 % speed and torque), seeds 0/1 | easy | hw_p0v84 | 0.111 / 0.050 | 100 / 47 % | 50 / 3 % |
| 40 × 22 mm pad | part | hw_p0pad40 | 0.096 | 100 % | 6 % |
| hips 60 / 70 mm (P08 reprint) | part | hw_p0s60 / s70 | 0.068 / 0.090 | 92 / 100 % | 26 / 54 % |
| C046 (1:147) roll/pitch/knee | servo | hw_p0c046 | 0.087 | 90 % | 13 % |
| **C046 + 60 mm hips**, seeds 0/1/2 | servo + part | hw_p0c046s60 | **0.148 / 0.168 / 0.174** | 100 % | 7 / 18 / 15 % |
| C044 (1:191) roll/pitch/knee (the knee's servo) | servo | hw_p0c044 | 0.121 | 100 % | 13 % |

- **Seed noise dominates small changes.** The same recipe gives 0.076 or 0.116 m/s on the as-built model,
  0.111 or 0.050 at 8.4 V. Nothing in the easy or part tiers beats the as-built spread.
- **Interaction:** with the slow C001 hips, hip spacing doesn't matter (0.068–0.090). With fast C046 hips,
  80 mm hips cap walking at 0.087 and 60 mm recovers 0.148–0.174 (round-2 sim: 0.157). The wider stance
  needs a bigger lateral weight shift every step; that only becomes the limit once the hips are fast enough.
- 60 mm hips are safe for walking: the C046+60 walker's knees touched in 2 of ~20 000 control steps
  (`scripts/leg_contacts.py`). The CAD collisions at 60 mm (both legs adducting together) are not used by gaits.

## E44 — Walking learned natively on the hardware (`hw_walk_scratch` → `hw_walk_scratch_stride` → `…_stride2`): **best walker**
- From scratch with the round-1 v5 `walk` recipe (DR, 80 M): 99 % reach 2 m, 0.071 m/s, **1.8 steps/s,
  start/stop 89 %, 1.1 A** — but it shuffles (1.5 m of sliding vs 0.9 m airborne per 20 s).
- Then `walk_stride_ft` (25 M) and +20 M: **98 % reach 2 m, 0.178 m/s at a 0.15 command, start/stop 98 %**,
  5.4 steps/s, airborne 5.3 m vs sliding 1.1 m per 20 s (real steps), RMS torque ≤ 0.83 × rated, 1.47 A mean.
  It tracks speed (0.04 / 0.13 / 0.23 m/s at 0.05 / 0.10 / 0.20) and stops at zero command (0.55 A standing).
- **Better than the round-2 sim walker (0.157 m/s, 44 % start/stop) and than any hardware change with the
  warm-start recipe — on the servos already bought.** The warm start was the bottleneck: the round-2 gait
  was shaped for faster, lighter legs and fine-tuning only bent it.
- Push-robust version (`hw_walk_best_push`, +20 M with 0.2 m/s kicks, rule → 10 M): walking under 0.2 m/s kicks
  every 2–4 s (`walkpush`, 40 s) 87 % alive / 92 % reach 2 m (before: 73 % / 86 %); start/stop 98 %; slower
  (0.135 m/s at a 0.15 command), 1.40 A.
- **Reproducibility: not reproduced.** Seed 1 (`hw_walk_scratch_s1`, 60 M; rule → 20 M) learned to step (6 steps/s,
  0.080 m/s, start/stop 81 %, 2.0 A) instead of shuffling; its stride fine-tune (`…_s1_stride`, from the 60 M final
  params, rule → 12 M) reached only 0.084 m/s, start/stop 48 %, 1358 overspeed events, knee RMS 1.04 × rated.
  Not a clean replica (60 vs 80 M, final vs best checkpoint as init), but the honest reading is: **the 0.178 m/s
  as-built walker exists, and training found it on 1 of 2 seeds.**

## E45 — Power (`scripts/power_audit.py`, details in `hardware/leg_p0/power.md`)
- Supply current from simulated torque × speed × duty through the servo motor model (no regen, +30 mA idle
  per servo, gearbox friction not modelled), 7.4 V, 20 episodes.
- Quiet stand 0.4–0.7 A; native walker 0.7–1.6 A mean rising with speed (1.4 A @ 0.15 m/s, 10 W), p99 4.7 A,
  20 ms peaks ~9 A; warm-started walkers 1.8–2.2 A regardless of speed (they step ~5/s even when standing).
  Knees are the hungriest servos (p99 ~3 A, peaks at the 3.8 A stall current).
- Sizing: ≥ 10 A peak / ≥ 3 A continuous supply, a bulk capacitor at the robot, one power branch per leg.

## Round 3 summary
**Run index** (23 runs, all completed; CPU gates 100 seeds each, rule = best in-training eval, step 0 excluded)

| # | runs | what | outcome |
|---|---|---|---|
| E40 | — | CAD-derived model of the as-built robot | 907 g; statics fine; hips 2× slower and 2.1× more effective inertia than the round-2 sim |
| E41 | `hw_stand_ft`, `hw_stand_scratch`, `hw_stand_push(2)` | quiet stand, then pushes | **98 % alive, 98 % no-lift, survives 0.3 m/s pushes** |
| E42 | `hw_walk_ft(_s1)`, `hw_walk_cad` | round-2 walker transferred | walks, slowly: 0.076–0.116 m/s, start/stop ~26 % |
| E43 | 12 runs | 8.4 V, pad 40, hips 60/70, C046, C044, C046+60 | only C046 + 60 mm hips is a clear gain (0.148–0.174, 3 seeds) |
| E44 | `hw_walk_scratch` → `_stride` → `_stride2`, `_s1…`, `hw_walk_best_push` | gait learned natively on the hardware | **0.178 m/s, 98 % start/stop on the as-built servos** (1 of 2 seeds) |
| E45 | — | power audit | walking 1.4–2.2 A mean, ~9 A 20 ms peaks; quiet stand 0.4–0.7 A |

**Answers**
1. *Is the as-built configuration possible?* **Yes.** It stands quietly (98 % no-lift), survives pushes, and
   walks. The best as-built walker (`hw_walk_scratch_stride2` @20 M) walks at 0.178 m/s with 98 % start/stop;
   the push-trained version (`hw_walk_best_push` @10 M) trades speed (0.135) for robustness.
2. *What limits it?* The C001 hips: half the speed of the sim's C046 and ~5× its reflected rotor inertia
   (estimate). Joint speeds hit the no-load limit while walking and recovering from pushes. The 1:191 knee is fine.
3. *What helps, cheapest first?*
   - **Training (free): learn the gait natively on the hardware model** (from scratch → stride fine-tune) rather
     than transferring the round-2 gait. Biggest effect seen (0.178 vs ~0.09 m/s), but found on 1 of 2 seeds.
   - **Settings/parts:** 8.4 V supply, a 40 mm pad, 60–70 mm hips: no effect beyond seed noise with the
     C001 hips. Narrower hips only pay off together with faster hip servos.
   - **Servos:** C046 on roll/pitch/knee **plus 60 mm hips** (a P08 reprint): 0.148–0.174 m/s on 3 of 3 seeds
     with the warm-start recipe — the most *reliable* gain, but it stops poorly (7–18 %). C046 alone (80 mm)
     gave nothing; C044 hips 0.121 (1 seed) at the highest current.
4. *Power:* see E45 and `hardware/leg_p0/power.md`: size for ~10 A peaks at 7.4 V, 2 A walking mean;
   one power branch per leg; knees are the hungriest servos and will run warm.

**Lessons (round 3)**
1. **Model the hardware you have, from the CAD.** A part-accurate model moved every conclusion: the
   reflected rotor inertia of a 1:345 gearbox outweighs the leg itself, and the 80 mm hips cost walking
   speed only once the hips are fast.
2. **Single seeds can't rank small changes.** The same recipe gave 0.076 or 0.116 m/s; 8.4 V gave 0.111 or
   0.050. Only effects larger than the seed spread (here ±0.02–0.04 m/s) count; three seeds settled the one
   real hardware effect.
3. **A transferred gait carries its old body's assumptions.** Fine-tuning the round-2 walker plateaued around
   0.09 m/s; a gait grown on the new model reached 0.178. Warm starts save time but cap the result when the
   dynamics change a lot.
4. **Shuffle first, then stride, worked once.** The from-scratch walker shuffled (efficient, stops well) and the
   stride reward turned it into real steps while keeping the stopping — but seed 1 took a different path.
5. **Power follows the gait, not the speed.** Walkers that step in place at zero command draw 4× a quiet stance;
   the native walker's current scales with speed. Peaks (~9 A over 20 ms) are the same for every gait.

**Next steps (not run)**
- More seeds of the native curriculum, initialized from the best (not final) checkpoint, to make the
  0.178 m/s result reliable.
- Measure on the real robot: knee stall torque (C044 spec is uncertain), servo current with an inline meter, and
  joint speeds under load — then re-fit `configs/actuators.yaml` and regenerate the model.
- If buying servos: C046 on roll/pitch/knee only together with narrower (60–70 mm) hips.

**GPU-time accounting (12:16 – 18:51, one job at a time):** 23 runs, about 6.6 h of the 7 h budget, all
completed. The queue runner's 20-minute stall watchdog never fired; `scripts/health_watch.sh` reported every
30 min (re-armed after each heartbeat). Evaluation (`scripts/eval_hw.sh`) ran on the CPU in parallel, so the
GPU never waited for it. One mid-run reprioritization (17:05) replaced four queued runs after E44's first result.

---

# Round 4 — walking that isn't a skitter (2026-10-29/30, `hw_p0_8dof`)

**Goal.** Every walker so far, including the round-3 best (`hw_walk_scratch_stride2`, 0.178 m/s), *skitters*:
5–7 touchdowns per second, 45–60 ms in the air, 1–3 cm of foot travel per step and ~5 mm of swing clearance.
Round 4 asks for a normal walk on the as-built (CAD-derived) model, and as a stretch an elegant, narrow,
"catwalk" one. 11 GPU-hours, one job at a time, same queue/watchdog as round 3.

**How gaits are judged.** Reward does not say whether a gait looks like walking, so round 4 adds
`scripts/gait_quality.py`: at a 0.15 m/s command with the heading-hold loop on, per step (touchdown after
≥ 60 ms of air) it measures air time, foot travel (lift-off → touchdown), swing clearance, step width; plus
the share of foot travel spent sliding, double-support fraction, pelvis sway, speed along the starting heading
(`vx`) and in the body frame (`body_vx`), and heading drift. `scripts/screen_r4.sh` ran it on every finished run.
Gates (walk40, startstop, …) are the usual 100-seed CPU evaluations.

## E46 — Reference gait for imitation (`slender_biped/ref_gait.py`)
- A kinematic walking pattern for the as-built leg, solved once by IK on the MuJoCo model into a lookup table
  over (forward speed command, gait phase): stance tip moves back at the walking speed, swing is a cycloid with a
  smooth 2–2.5 cm lift, lateral sway from the linear inverted pendulum's periodic orbit, optional narrower track
  and pelvis twist. Zero command = the standing pose. Step length = speed × stance time, so cadence and duty
  factor are parameters: tables `ref30/36/42` (3.0/3.6/4.2 steps/s, duty 0.55), `d65_30` (duty 0.65), `nar30/36`
  and `d65_w60` / `d65_cat` (40–60 mm tracks), `d65_30_s60` (60 mm-hip model).
- The table is checked against the servos when built: peak joint speed ≤ 0.8 × no-load at 0.15 m/s for the tables
  used (the 3.6 steps/s duty-0.65 table needed 0.9 and was not used).
- Env support (both the MJX trainer and the CPU evaluator): `ref_table`, `ref_residual` (action = correction around
  the reference instead of around the standing pose, action scale 0.25), imitation rewards `ref_q` (joint
  tracking), `ref_tip` (tip height), `ref_contact` (contact schedule); `ref_yaw_weight` (hip-yaw joints in
  `ref_q`); `width` (penalty on lateral tip separation above `width_target`); `tip_condim` (E51).

## E47 — First imitation runs: real steps, but "marching in place" (`r4_ref30_res`, `r4b_*`)
- `walk_ref` (3.0 steps/s, residual): from scratch it learned real stepping — **220 ms air time, 22 mm clearance**
  — but at a 0.15 m/s command it moved 0.005 m/s. Cause: with `track_sigma` 0.02, standing still still earned 32 %
  of the tracking reward while the imitation terms paid in full. Stopped at 30 M.
- `walk_ref2` (sharper/heavier tracking): same picture at 60 M (0.015 m/s along the heading).
- **Fine-tuning the skitter with imitation rewards keeps the skitter** (`r4b_ft_abs`, from the round-3 best, 40 M):
  0.14 m/s, but still 60 ms air, 2.2 cm, 5 mm clearance. A converged gait does not move to a loose imitation term.

## E48 — Harder rules from scratch don't learn (`r4c_*`, `r4d_*`)
- Tight joint kernel (`ref_sigma_q` 0.02) + slip −30 (`walk_ref3`): 52-step episodes at 10 M from scratch; as a
  fine-tune of the marcher it kept marching (0.02 m/s).
- `walk_ref2` + slip −30 + easier speed range (`walk_ref4`): also 60–70-step episodes at 10 M, on both the normal and
  a 40 mm "catwalk" track. Leg–leg collision in the narrow reference was ruled out (no contacts in the table).
  **Slip −30 blocks learning from scratch; the narrow track is not the cause.** Runs stopped at 10 M.

## E49 — The "marcher" was walking in circles
- A rollout in the MJX training env itself gave 0.094 m/s for the same policy that the CPU screen measured at
  ~0. Not a simulator mismatch: without the heading-hold loop the CPU also sees 0.04–0.13 m/s. **The policy walks
  forward in its body frame (0.08–0.09 m/s) but turns 3–4°/s**; progress along any heading is ~0.
  (`gait_quality.py` now reports `body_vx` and heading drift.)
- Mechanism: with duty 0.55 it is 80 % of the time on one small, edge-loaded pad, which gives almost no grip in yaw,
  so every swing twists the body. It spun ~160° while starting to walk and then ignored the saturated turn
  command. The skitter steers fine because it keeps both pads down half the time.
- Stronger heading rewards (`walk_ref5`, heading 3, yaw-rate 2): drift 1–4.5°/s, not fixed. Freeing the hip-yaw
  joints from `ref_q` (`walk_ref6`): not fixed either (4.8°/s).

## E50 — More double support: **straight walking with real steps** (`r4h_d65_30_res`, seed 1 `r4h_d65_30_s1`)
- Reference with duty 0.65 (both pads down ~30 % of the cycle; swing still 0.23 s at 3 steps/s), `walk_ref6`.
- Seed 0 @60 M: heading held (0.26°/s), 0.142 m/s along the heading, 5.5 cm foot travel, 120 ms air, 16 mm clearance.
  100-seed gates: walk40 99 % reach 2 m, 0.164 m/s; **but** start/stop 51 %, it steps in place at zero command, the hip
  roll/pitch servos run at ~1.0 × rated RMS torque and there were 7 513 overspeed events.
- **Reproduced** on seed 1 (@50 M: 0.117 m/s straight, 5.9 cm, 160 ms air, 19 mm clearance) — unlike the round-3 best
  walker (1 of 2 seeds).

## E51 — Pad yaw friction was switched off in the model (`tip_condim`)
- The CAD-derived model gives the pad a torsional friction coefficient (0.004), but the tip contacts used
  `condim="3"`, which ignores it: yaw resistance came only from box-corner contacts. `tip_condim=4` turns it on
  (MJX and CPU). This is a modelling correction, not a tuning knob: the old skitter is unaffected (still holds
  heading within 3°, 0.16–0.18 m/s).
- Effect on the 0.55-duty marcher (fine-tune `r4i_ft_c4`): still wanders (0.02 m/s along the heading) — pad grip
  alone is not enough; the duty factor is the main lever.
- Duty 0.65 + condim 4 from scratch (`r4j_d65_30_c4` @60 M, 100 seeds): walk40 **100 %** reach 2 m, 0.130 m/s,
  start/stop **84 %** (vs 51 %), overspeed events **715** (vs 7 513), max RMS/rated 0.98.
- 3.6 steps/s, duty 0.55, condim 4 (`r4i_ref36_c4` @60 M): also walks straight-ish (0.4°/s), 0.109 m/s, and is the
  calmest gait of the round: 3.7 steps/s, 200 ms air, 5.9 cm, 20 mm clearance, 14 % sliding — but a 119 mm stance.

## E52 — Polishing, and standing by handing over to the stander (`r4j_d65_c4_cont`, `r4l_polish`, `scripts/switch_eval.py`)
- Continuing the condim-4 walker (+40 M, rule → 20 M): 7.6 cm foot travel, 0.11 m/s; gates: 100 % reach 2 m,
  0.134 m/s, start/stop 71 %, max RMS/rated 0.98, 2.15 A mean / 8.95 A 20 ms peak.
- `walk_ref7` polish (quiet-stance terms at zero command, 2× torque/overspeed pressure, 30 M, rule → 20 M):
  overspeed events 2 588 → 1 590, otherwise the same (start/stop 68 %, hip roll still ~0.97 × rated).
  **It still steps at zero command** (≈ 290 lift-offs in 30 s) — the same wall as rounds 2–3: quiet stance only
  came from a dedicated standing policy.
- So stand with the stander: `switch_eval.py` runs the walker while the command is non-zero and hands over to
  `hw_stand_push2` (E41) after the command drops to zero; walk 6 s → stop 6 s → walk 6 s, 100 seeds:

| walker (+ `hw_stand_push2`) | success | falls | quiet stop (no lift-off, last 3 s) | walked per phase |
|---|---|---|---|---|
| `r4j_d65_30_c4` @60 M, 0.3 s hand-over delay | **88 %** | **0 %** | 99 % | 0.55 / 0.61 m |
| `r4k_cat2_ft` @30 M (E53), 0.0 / 0.3 / 0.6 / 1.0 s | 78 / 85 / **85** / 85 % | 12 / 7 / **5** / 5 % | 83 / 91 / 92 / 91 % | ~0.71 m |

  Residual sway after the hand-over is 0.022 m/s RMS (the stander alone: 0.015). Handing over in a double-support
  window of the walker's clock changed nothing here (the schedule is deterministic, so every stop already lands in
  one). The single walking policies manage only 15–84 % start/stop and never stand still.

## E53 — Elegant "catwalk" walking by curriculum (`r4k_w60_ft` → `r4k_cat_ft` → `r4k_cat2_ft`)
- A narrow track could not be learned from scratch (E48), so narrow it step by step from the condim-4 walker, each
  stage 30 M, lr 1e-4: 60 mm reference track → `d65_cat` (45 mm track, ±4° pelvis twist through the hip-yaw
  joints, 30 mm lift with a rounder swing) → the same plus a penalty on lateral tip separation above 60 mm
  (`walk_ref8`; the hip-roll correction kept pushing the feet back out).
- Measured step width (lateral gap between consecutive touchdowns): **116 mm** (condim-4 walker, wider than the
  80 mm hips) → 109 → 98.5 → **72 mm** (inside the hip width).
- `r4k_cat2_ft` @30 M is the best walking result of the round: 100 % reach 2 m in 40 s at **0.162 m/s**, 6.2–6.4 cm
  foot travel, 160 ms air, 19 mm clearance, **9–10 % sliding**, heading drift 0.1°/s. Start/stop on its own 15 %
  (it does not slow down; use the hand-over, E52). Max RMS/rated 1.0 (hip roll), overspeed 3 077.

## E54 — Push robustness costs the gait (`r4m_cat_push`, 0.2 m/s kicks, 20 M from E53, rule → 10 M)
- Walking under 0.2 m/s kicks every 2–4 s (`walkpush`, 40 s, 100 seeds): alive 53 → 62 %, reach 2 m 72 → 77 %.
- But the gait slid back toward a skitter: 6.2 steps/s, 100 ms air, 2.9 cm, 12 mm clearance. Not adopted; the
  round-3 push walker (`hw_walk_best_push`, 87 % alive) remains the robustness reference.

## E55 — Cadence with pad yaw friction on (`r4i_ref36_c4`, `r4i_ref42_c4`, duty 0.55, from scratch, 60 M)
- With condim 4, faster references also walk straight without the 0.65 duty factor:

| reference | steps/s (measured) | air | foot travel | clearance | sliding | sway | width | vx (heading) |
|---|---|---|---|---|---|---|---|---|
| 3.6 steps/s | 3.7 | 200 ms | 5.9 cm | 20 mm | 14 % | 46 mm | 119 mm | 0.109 |
| 4.2 steps/s | 4.2 | 200 ms | 6.4 cm | 18 mm | 11 % | 42 mm | 124 mm | 0.128 |

- These hold their reference cadence (the duty-0.65 walkers drift up to 4.4–5.4 steps/s) and have the calmest
  bodies (least sway), but a wide stance.

## E56 — Reproducibility and the last hardware questions
- **Seed 1, duty 0.65:** condim 3 (`r4h_d65_30_s1` @50 M: 0.117 m/s straight, 5.9 cm, 160 ms air) and condim 4
  (`r4j_d65_30_c4_s1` @40 M: 0.109 m/s straight, 6.6 cm, 180 ms air, 19 mm clearance). **2 of 2 seeds** for both.
- **Seed 1, style curriculum** (`r4k_w60_s1` → `r4k_cat_s1` → `r4k_cat2_s1`, from the seed-1 condim-4 walker):
  step width 114 → 92 → 96.5 → **73 mm** (seed 0: 116 → 109 → 98.5 → 72); final @30 M: 0.144 m/s straight,
  4.7 cm, 160 ms air, 19 mm clearance, 10 % sliding. **Reproduced.**
- **60 mm hips** (`r4n_s60_d65`, same recipe on `hw_p0s60_8dof` with a re-solved reference): longest steps of the round
  (8.4 cm, 180 ms air), 99 % reach 2 m at 0.135 m/s, start/stop 65 %, overspeed 581 — but hip roll still 0.95–0.99 ×
  rated and 2.0 A mean. **Narrower hips do not cool the hips.**
- **Stronger torque penalties** (`r4o_cat_cool`, torque −0.4, over-rated −6, overspeed −4, 30 M from E53): hip roll
  0.98–1.04 × rated, unchanged; the early checkpoint slid toward faster, shorter steps. With the C001 hip servos a
  real-step gait at ~0.15 m/s needs about rated torque at the hips.
- **Power** (`scripts/power_audit.py`, 20 episodes): real steps cost ~2.2 A mean (16 W) vs 1.5 A for the skitter;
  20 ms peaks ~9 A as before, with one rare 14.6 A event for the catwalk walker (99 % of 20 ms windows < 5.5 A).
  `hardware/leg_p0/power.md` updated.

## Round 4 summary
**Run index** (28 runs; 22 completed, 6 stopped early once their question was answered; gait metrics from
`scripts/gait_quality.py` at a 0.15 m/s command, 3–5 seeds; gates 100 seeds)

| # | runs | what | outcome |
|---|---|---|---|
| E46 | — | reference gait by IK on the CAD model; imitation + residual actions | tables for 3.0–4.2 steps/s, duty 0.55/0.65, 40–80 mm tracks |
| E47 | `r4_ref30_res`, `r4b_*` | imitation from scratch; skitter fine-tune | real steps (220 ms air) but no progress; skitter fine-tune stays a skitter |
| E48 | `r4c_*`, `r4d_*` | tight kernel, slip −30, narrow track from scratch | does not learn |
| E49 | `r4f_*`, `r4g_*` | heading rewards, free hip yaw | the "marcher" walks in circles; rewards don't fix it |
| E50 | `r4h_d65_30_res`, `_s1` | duty 0.65 | **straight walking with real steps, 2/2 seeds** |
| E51 | `r4i_ft_c4`, `r4j_d65_30_c4`, `_s1` | pad torsional friction on (model fix) | start/stop 51 → 84 %, overspeed ÷10; 2/2 seeds |
| E52 | `r4j_d65_c4_cont`, `r4l_polish` | polish; stand via hand-over | still steps at zero command; walker + stander 85–88 % walk/stop/walk, 0–5 % falls |
| E53 | `r4k_w60_ft` → `r4k_cat_ft` → `r4k_cat2_ft` | catwalk curriculum + width penalty | **72 mm stance, 6.2 cm steps, 0.162 m/s**; reproduced on seed 1 (73 mm) |
| E54 | `r4m_cat_push` | push training | more robust, gait reverts toward a skitter |
| E55 | `r4i_ref36_c4`, `r4i_ref42_c4` | faster references, condim 4 | straight, calmest gaits (3.7/4.2 steps/s, 200 ms air), wide stance |
| E56 | `r4n_s60_d65`, `r4o_cat_cool`, seed-1 replicas | 60 mm hips, torque pressure, replication | hips stay at rated torque; results reproduce |

**Answers**
1. *Can it walk normally instead of skittering?* **Yes, in simulation on the as-built model.** Foot travel 2.4 → 6–7 cm per
   step, air time 60 → 160–200 ms, clearance 5 → 17–20 mm, sliding 22 → 9–14 %, straight, 0.13–0.16 m/s, on 2 of 2 seeds.
2. *Elegant?* A curriculum narrows the walk to a 72–73 mm "catwalk" line (the skitter: 189 mm) with a slight pelvis
   twist and a higher swing (2/2 seeds).
3. *What it costs:* ~45 % more current (2.2 A mean), hip servos at rated torque (warm hips), and stopping needs the
   hand-over to the round-3 stander. Push robustness is below the round-3 push walker.

**Lessons (round 4)**
1. **Measure the gait, not the reward.** The decisive bug ("marching in place" was walking in circles) was invisible in
   reward and in progress-along-heading; body-frame speed and heading drift exposed it in minutes.
2. **Imitation shapes a gait from scratch; it does not reshape a converged one.** Fine-tuning the skitter with imitation
   rewards kept the skitter; push training on the new gait pulled it back toward one.
3. **For a footless leg, double support is the steering wheel.** On one small, edge-loaded pad there is almost no yaw
   grip; reward terms could not buy heading control that the contact physics did not offer. Changing the gait (duty
   factor, cadence) did.
4. **Check the model's defaults.** A torsional friction coefficient sat in the CAD-derived model for a whole round, inactive
   because of `condim="3"`. Turning it on changed start/stop by 33 points and did not affect the old results.
5. **Curricula for style.** A narrow track from scratch never learned; three short fine-tunes from a working walker did,
   and reproduced.

**GPU-time accounting (22:27 – 07:58, one job at a time):** 28 runs, about 9.5 h of the 11 h budget. The queue runner's
stall watchdog never fired; `scripts/health_watch.sh` reported every 30 min. Six runs were stopped by hand as soon as
their question was answered (marching, no learning, known heading flaw). Evaluation ran on the CPU in parallel.
