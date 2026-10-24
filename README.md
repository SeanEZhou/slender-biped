# Slender-leg biped: footless walking under hobby-servo limits

A 38 cm, 0.85 kg biped with narrow hips, tubular legs and **no feet**, just a 16 mm rubber ball at
the end of each shin (inspired by the Elster / Aigis silhouette). Eight off-the-shelf Feetech
servos (~$210 in total) drive it, each modelled with its real torque-speed curve, latency and
quantization. It is trained with GPU reinforcement learning (MuJoCo MJX + Brax PPO) and checked in
an independent CPU simulator.

**Questions:**
1. Can this footless form factor stand and walk on hardware a hobbyist can buy?
2. Is being footless what makes it unable to stand still, or is something else to blame?

**Answers, in simulation:**
1. Yes. It stands indefinitely, walks straight and stops on command, with every servo below its
   rated torque.
2. For standing, yes. Every flat foot we tried, with or without an ankle, stood without stepping;
   the ball tip never did. A **30 × 18 mm flat pad at the shin tip**, which keeps the ball-tip
   silhouette, *can* stand without stepping, though training found that on 1 of 2 seeds. For
   walking, feet were not the fix: with feet the policies skated or marched in place instead of
   taking real strides.

> **Status (2026-10-23): simulation only.** No hardware has been built yet. All numbers come from
> held-out evaluation seeds. Simulated feasibility is not validated hardware capability.

<table>
<tr><th>Stand<br><sub>zero command</sub></th><td><img src="assets/gifs/stand.gif" width="520"></td></tr>
<tr><th>Walk<br><sub>0.15 m/s command, heading hold</sub></th><td><img src="assets/gifs/walk.gif" width="520"></td></tr>
<tr><th>Walk → stop → walk</th><td><img src="assets/gifs/start_stop.gif" width="520"></td></tr>
</table>

<sub>Ball tip, one policy (`cadence_A8_ft`). All clips use evaluation seed 10000 (the first seed, not
cherry-picked), rendered in real time by <code>scripts/render_gif.py</code>.</sub>

## Results: ball tip (the selected design)

One policy controls all three behaviours. It was selected by a rule declared in advance (highest
in-training evaluation reward), not by its test results. Each gate uses 100 held-out seeds in the
CPU simulator, with the full contact model and the servo model in the loop.

| gate | target | result |
|---|---|---|
| free stand ≥ 15 s | ≥ 70 / 100 | **100 / 100** (also 100 / 100 at 30 s) |
| walk ≥ 2 m without falling | ≥ 70 / 100 | **100 / 100** in a 40 s test: median 5.3 m, 0 falls, 1.5° heading drift |
| walk → stop → walk | ≥ 50 / 100 | **98 / 100** (speed 0.14 → 0.01 → 0.14 m/s) |
| servo load | within ratings | sustained RMS torque **0.46–0.73 × vendor rated** on every joint; peak joint speed 0.79 × no-load; duty saturated 0.03 % of the time; 0 over-speed events |
| geometry | hip ≤ 24–28 % of height, shin ≤ 9–12 % | hip 28.4 % (just over the target), shin **5.8 %**; every servo fits inside its shell with a 1.2 mm wall |

This policy came out of four rounds of fine-tuning (v5 → v6b → stride → cadence). Compared with the
round-1 policy v6b, it walks faster (0.134 vs 0.086 m/s at a 0.15 command) and stops reliably
(98 % vs 79 %), with less servo load (0.73 vs 0.82 × rated).

**Not solved yet:**
- **Never still:** it stands by micro-stepping at about 6 steps/s.
- **Tiny steps:** it walks in about 2 cm steps at about 7 steps/s.
- **Turning:** it follows heading only through an outer IMU-yaw loop.
- **Pushes and slippery floors:** both remain weak, as measured on the round-1 policy.

## Feet control experiment: is footlessness the problem?

We gave the same frame feet in a separate model (baseline A is untouched), trained it with the
identical recipe, and compared. All rows: 100 held-out seeds. "No-lift stance" = the robot survived
30 s and no tip or sole ever left the floor after a 2 s settle.

<table>
<tr><th>Footed, 60 mm soles: stands</th><th>Footed: "walks" by skating</th></tr>
<tr>
<td><img src="assets/gifs/footed60_stand.gif" width="390"></td>
<td><img src="assets/gifs/footed60_skate.gif" width="390"></td>
</tr>
<tr><th>Flat pad 30 mm: stands</th><th>Flat pad 30 mm: walks</th></tr>
<tr>
<td><img src="assets/gifs/pad30_stand.gif" width="390"></td>
<td><img src="assets/gifs/pad30_walk.gif" width="390"></td>
</tr>
</table>

**Standing, same quiet-stand reward from scratch:**

| contact | alive @30 s | no-lift stance | lift-offs per run | steps/s |
|---|---|---|---|---|
| ball tip (16 mm) | 97 % | **0 %** | 210 | 6.1 |
| ball tip, ideal control (100 Hz, no latency, perfect sensors) | 92 % | **0 %** | 475 | — |
| flat feet with ankles: 40 / 60 / 90 mm | 98 / 96 / 92 % | **94 / 95 / 90 %** | 0 | ≈ 0 |
| flat 60 mm sole, no ankle | 95 % | **94 %** | 0 | 0.07 |
| flat pad at the shin tip, 20 × 14 mm | 95 % | 0 % | 210 | 5.3 |
| **flat pad at the shin tip, 30 × 18 mm** (seed 0) | 94 % | **85 %** | 0 | 0.03 |
| flat pad 30 × 18 mm, seed 1 (75 M steps) | 100 % | 0 % | 98 | 0.6 |

**Walking:**

| | ball (best) | footed 60 mm, ball recipe | footed 60 mm, stride recipe | pad 30 mm |
|---|---|---|---|---|
| reaches 2 m in 40 s | 100 % | 83 % | 0 % | 100 % |
| speed at a 0.15 m/s command | 0.134 m/s | 0.054 | 0.054 | **0.157** |
| steps/s | 7.2 | ≈ 0 (skates) | 5.3 (marches in place) | **4.25** |
| step length | 1.8 cm | — | ≈ 0 | **3.6 cm** |
| start → stop → start | **98 %** | 43 % | 16 % | 44 % |

What the control shows:
- **Standing still needs a flat contact, not an ankle.** Every flat foot stopped the stepping, even
  with no ankle. The ball tip never stopped stepping under any of four reward and control variants,
  including the ideal-control upper bound. Two point contacts leave a tip-over direction that no
  joint can resist, so stepping is the only strategy RL found.
- **The contact must be big enough.** A 20 mm pad still stepped. A 30 mm pad, which still fits the
  slender silhouette, stood with zero lift-offs in 85 % of runs. That was on one training seed; a
  second seed was still stepping (though slowing down) after 75 M steps. The pad makes quiet
  standing possible, but training does not reliably find it yet.
- **Feet did not produce strides.** Under the ball's walking reward the footed robot skated its
  soles forward. Under a stride reward it marched in place. Short steps are mostly a reward and
  training problem, not a foot problem.
- **The flat pad doubled step length** (3.6 vs 1.8 cm) at the same reward and initialization, and
  walked faster. Its walking policy, though, still steps in place at zero command and stops poorly
  (44 %). Combining the pad's quiet stance and its walking in one policy is untested.

## Key learnings

1. **Servo speed, not torque, is the binding limit.** The 12 V STS3215 has twice the torque, but
   its 45 rpm knee allows only about 1 step/s, so it was rejected before training. Its 7.4 V
   sibling (106 rpm, same case) works. A slimmer Dynamixel hip set stood (91 %) but went over
   speed 2,046 times per 100 runs, at ~3.4× the cost.
2. **Point contacts force stepping; a small flat pad can remove it.** See the control experiment above.
3. **Hardware plausibility has to be trained in.** Without a sustained-torque penalty, standing ran
   hip roll at 1.29 × rated torque. With it, ≤ 0.88 ×, and survival *rose* from 96 % to 99 %.
4. **Fine-tuning beats re-training for walking.** From-scratch stride rewards failed on the ball tip
   (it fell within 3 s). Fine-tuning a working walker step by step kept improving speed, stopping
   and servo margins. Warm starts must restore the critic: one fine-tune with a fresh critic
   collapsed within 10 M steps.
5. **Measure the gait, not just the reward.** The footed walker reached 2 m in 83 % of runs by
   skating, not stepping. Splitting foot travel into airborne versus sliding
   (`scripts/gait_anatomy.py`) exposed it.
6. **Declare checkpoint selection before testing.** Long runs regressed or collapsed after their
   peak (up to ~80 % loss of episode length). On flat feet the untrained policy even scored highest,
   so the rule was amended (excluding step 0) before any result of that run was seen.
7. **Traction and pushes, not strength, are the next bottleneck.** +20 % mass and −25 % servo
   strength barely matter. Low floor friction, doubled sensor noise and large pushes break walking.

## The design

<p>
<img src="assets/renders/baseline_A_8dof/three_quarter.png" height="340">
<img src="assets/renders/baseline_A_8dof/joint_axes.png" height="340">
</p>

- **Per leg:** hip yaw (STS3032), hip roll, hip pitch and knee (STS3215-C046, 7.4 V); 8 DoF total.
  The yaw, roll and pitch axes meet near one point, so the hip acts like a ball joint.
- **Knee drive:** the knee servo sits in the thigh and drives the knee through a 1:1 belt, so the
  knee ball stays slim.
- **Dimensions:** thigh 138 mm, shin 150 mm, centre of mass 27 cm above the floor.
- **Shells:** the printed shells are a parametric superellipse shape, generated from
  `configs/geometry.yaml`. The packaging check confirms every servo case fits inside its shell.
- **Mass:** the simulator gets each part's mass and inertia (servos, shells, electronics), so the
  centre of mass and balance match the CAD estimate.
- **Control frames:**
  - `footed_*` adds ankle pitch and roll with strong servos, plus flat soles. It is a test rig, not a
    design candidate.
  - `baseline_P30_8dof` swaps the ball for a 30 × 18 mm rigid rubber pad. It is the same frame
    otherwise.

## How it works

- **Simulator:** MuJoCo 3.17. Training runs on the GPU with MJX and a lean contact set (4,096
  parallel environments). Evaluation runs in an independent CPU simulator with every contact pair
  and a more accurate solver.
- **Servo model:** in [`slender_biped/actuator.py`](slender_biped/actuator.py).
  - Duty is clipped, then torque falls linearly with speed, as in a DC motor
    (stall torque × (1 − ω/ω₀)), with a current limit.
  - It also models 4 ms bus latency, slew limiting, a position deadband, quantization and reflected
    rotor inertia (J·N²).
  - Saturation and over-speed are logged, never hidden.
- **Policy:** PPO with an asymmetric actor-critic, running at 50 Hz with 2 ms physics steps.
  - The actor sees only signals a real robot has: IMU gravity and gyro, joint positions and
    velocities, last action, command and gait clock.
  - The critic also sees privileged simulator state.
- **Rewards:**
  - velocity and heading tracking, a stop reward, upright and height terms;
  - gait shaping, plus a stride reward and a lift-off penalty added in round 2;
  - hardware penalties for sustained torque, saturation and over-speed.

  Details are in [`slender_biped/mjx_env.py`](slender_biped/mjx_env.py) and the task presets in
  [`scripts/train.py`](scripts/train.py).
- **Domain randomization:** tip friction, link masses, CoM, rotor inertia, gear friction, servo
  strength, speed and gain, latency, and backlash.
- **Evaluation:**
  - A NumPy re-implementation of the policy, identical to Brax inference (max |Δa| = 0). Seeds 10000+
    are never used in training.
  - Stillness and gait metrics: lift-offs, tip travel, airborne versus sliding foot travel, and step
    length.

## Round-1 ablations (ball tip)

Standing unless noted; 100 held-out seeds each.

| question | change | result |
|---|---|---|
| tip shape | 32 × 13 mm rocker vs ball | 99 % vs 99 % alive at 30 s; walking 92 % vs 85 % reach 2 m. **No difference.** |
| thermal margin | sustained cap at 0.25 × stall | 95 % alive. **Stands under a pessimistic heat assumption.** |
| servo family | slim Dynamixel hips (24 % hip width) | 91 % alive, 2,046 over-speed events. **Speed-limited, ~3.4× cost.** |
| robust training | sensor noise ×1.5 + pushes | noise ×2 robustness 0.66 → 0.92; pushes not improved. **Trade-off.** |
| domain randomization | DR fine-tune | no robustness gain |
| reproducibility | training seed 1 vs seed 0 | 97 % vs 99 % alive at 30 s |

Walking robustness of the round-1 policy v6b (30 seeds per condition; share reaching 1 m in 20 s):

| nominal | mass ×1.2 | strength ×0.75 | 5° slope | latency 12 ms | sensor noise ×2 | friction 0.5 | push 0.3 m/s | friction 0.35 | push 0.5 m/s |
|---|---|---|---|---|---|---|---|---|---|
| 100 % | 100 % | 100 % | 100 % | 93 % | 80 % | 67 % | 67 % | 27 % | 27 % |

The full experiment trail, ~40 training runs including the failures, corrections and dead ends, is
in [`docs/experiments.md`](docs/experiments.md).

## Reproduce

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python models/generate_model.py && .venv/bin/python scripts/regen_train_models.py
.venv/bin/python -m pytest -q tests

# evaluate the included policies on CPU (100 held-out seeds)
.venv/bin/python scripts/evaluate.py cadence_A8_ft --test walk40 --params params_0020M.pkl
.venv/bin/python scripts/evaluate.py q2_stand_P30 --test stand --params params_0022M.pkl
.venv/bin/python scripts/gait_anatomy.py stride_P30_ft --params params_0020M.pkl
.venv/bin/python scripts/render_gif.py cadence_A8_ft --params params_0020M.pkl --test walk40 --seconds 12 --out walk.gif

# train on a GPU (one job at a time; gpu_python.sh hides a system CUDA from LD_LIBRARY_PATH)
scripts/gpu_python.sh scripts/train.py --name my_walk --model baseline_A_8dof --task walk --steps 150e6 --dr
```

`runs/` keeps every run's config and learning curve. Weights are included only for the selected
policies and the checkpoints they were fine-tuned from.

## Repository map

| path | contents |
|---|---|
| `configs/` | parametric morphology (`geometry.yaml`, including the control frames); servo catalogue with sources and assignment sets (`actuators.yaml`) |
| `models/generate_model.py` | MJCF generator: exterior shells, servo envelopes, collision primitives, full and training contact sets |
| `slender_biped/` | servo model, MJX training env, CPU evaluator, morphology and statics tools, renderer |
| `scripts/` | training, evaluation, gait anatomy, robustness sweeps, torque audit, renders, and the one-job-at-a-time GPU queue with stall and health watchdogs |
| `results/` | evaluation JSONs for every reported number; plots |
| `docs/` | [spec](docs/project_spec.md), [morphology](docs/morphology.md), [actuator research](docs/actuator_research.md) and [feasibility](docs/actuator_feasibility.md), [experiment log](docs/experiments.md), [sim-to-real risks](docs/sim2real_risks.md) |

## Limitations

- **Servo specs:** Feetech figures come from retailer listings. Rotor inertia, gear friction and
  the "rated" torque are estimates, and thermal behaviour is unverified.
- **Contact model:** contacts are MuJoCo soft constraints. Real rubber tips and pads, floor
  compliance and belt backlash will differ. The pad's benefit depends on its edges staying in
  contact, which real compliance could help or hurt.
- **Single seeds:** most walking results and the footed controls come from one training seed each.
  Several runs collapsed after their peak and were used at the rule-selected checkpoint.
- **Control-frame confound:** the footed frame is 250 g heavier (strong ankle servos).
- **Post-hoc result:** one early walking result (round 1, v5 at 30 M) was picked after looking at
  test results; it is labelled as post hoc in the log.
- **Next steps:**
  - make the 30 mm pad's quiet stance reproducible (seeds, gentle-start curriculum);
  - one policy that both walks and stands still on the 30 mm pad;
  - longer strides through motion priors or reference gaits;
  - explicit turning and push recovery;
  - bench tests of the servo torque-speed curve, tip friction and IMU noise.

  [`docs/sim2real_risks.md`](docs/sim2real_risks.md) lists the risks.
