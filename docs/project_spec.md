# Project Specification: Slender-Leg Biped — Elster / Aigis-Inspired Locomotion POC

**Status:** Initial proof of concept (simulation first)  
**Primary audience:** Autonomous coding agent implementing the project  
**Goal:** Design, simulate, and train an independently balancing/walking biped lower body with the slender, tubular, minimal-foot silhouette associated with Elster (*SIGNALIS*) and Aigis (*Persona 3*), constrained to physically plausible, commercially accessible actuators.

## 0. Operating instructions for the coding agent

Build a working, reproducible prototype rather than just a design proposal. Make sensible engineering choices, document them, and proceed iteratively. Prefer a simple tested baseline over a complex unvalidated architecture. Keep all major morphology, actuator, control, and training parameters configurable. **Never silently replace torque-limited actuators with unrealistically powerful ideal joints to make learning succeed.** Record failed experiments as well as successful ones. Do not claim physical feasibility on the basis of an RL reward curve alone.

**Deliverables:** executable MuJoCo robot model; leg and foot geometry; actuator selection/constraint sheet; standing and locomotion training pipelines; trained policies/checkpoints when compute allows; evaluation harness; plots/videos; sim-to-real risk review; attractive portfolio README. Only claim completed experiments that actually ran.

## 1. Research question and hypotheses

**Core question:** Can an affordable, modestly scaled biped with a narrow, visually unobtrusive pelvis and elegant near-cylindrical legs learn to balance and walk **without broad, human-like feet**?

Test two lower-leg contact geometries using otherwise identical mass, actuators, joint topology, training budget, and initializations:

- **A — stick/point foot:** tapered leg ending in a small rounded, high-friction, replaceable rubberized tip (finite radius for stable physical modeling; no magical zero-area contact).
- **B — curved micro-foot:** elongated narrow rounded/rocker-like tip with a small fore–aft contact footprint, integrated seamlessly into the shin. No conspicuous heel/toe/sole. Radius, arc and length parameterized.

Hypotheses: both may achieve dynamic walking, but B may make stationary balance, starts/stops, and pushes more manageable. This is a hypothesis, not a result. Evaluate whether either can stand without stepping; do not hide an inability to stand by showing only walking videos.

## 2. Aesthetic requirements — non-negotiable

Character references are **visual inspiration, not a request to copy exact art assets**. Preserve:

1. A narrow, anatomically plausible hip/pelvis width, not a broad box containing a dozen exposed servos.
2. Visually continuous, slender, mostly tubular thighs and shins; subtle tapering is welcome.
3. A long-leg impression, especially the lower legs; minimal external linkages or bulky gearbox housings.
4. No visually separate conventional feet. Feet must be literal rounded tips (A) or very narrow curved contact patches visually continuing the leg (B).
5. Bilateral symmetry, clean protective joint covers, and simple, printable mechanical parts.

Allow a tiny instrumented pelvis/test mass in place of a torso; there is no requirement to model head, arms, hands or entire humanoid. Do **not** stabilize the robot with an invisible torso, fixed base, hidden floor constraints, an external spring, or an unrealistically large contact patch. Supports/harnesses are permitted **only for explicitly identified curriculum stages**, disabled during final free-standing evaluations.

**Appearance review gate:** Save full-body orthographic front/side renders and a dimensioned silhouette. Report hip width / standing height, maximum shin thickness / shin length, and effective ground-contact footprint. Before optimization, enforce aspirational ratios (tune based on actual motor packaging): pelvis width <= ~24–28% of robot height; shin visible width <= ~9–12% of robot height. These are initial visual design targets, **not** proven mechanical constraints; explain deviations. Differentiate *exterior silhouette* from *internal actuator envelope*.

## 3. Proposed scale and baseline morphology

**Preferred first attempt:** standing pelvis-to-ground height approximately **0.30–0.35 m**, target assembled lower-body mass **~0.6–1.0 kg**, adjustable following CAD/actuator mass accounting. Height is intentionally not 1:1. If required hip/knee motors cannot physically fit or pass torque tests, propose a **smaller** variant rather than strengthening motors without checking cost/size.

Start with a floating, unconstrained pelvis/base (6-DoF free joint) plus:

| Joint per leg | Count | Function |
|---|---:|---|
| Hip pitch | 1 | forward/backward swing |
| Hip roll | 1 | lateral weight transfer |
| Hip yaw | 1 | turning / foot orientation |
| Knee pitch | 1 | swing clearance, stance extension |
| Ankle pitch | 1 | limited distal-foot/rocker orientation, if packaging permits |
| **Total** | **10** | 5 actuators/leg |

Treat **8 DoF (hip pitch+roll+yaw, knee pitch per leg; passive fixed tip)** as an explicitly tracked lower-cost ablation if ankle motors worsen appearance/weight. Ankle roll can be considered later, but do not add joint count without showing why. Model actual motor locations, shafts, bearings, clearances, and mechanical limits; a mathematical joint does not prove a plausible linkage.

Initial reference proportions (tunable): hip-center spacing 70–90 mm; thigh 130–155 mm; shin 140–165 mm; small contact-tip radius 5–10 mm (variant A); curved micro-foot length 20–40 mm and width 10–20 mm (variant B). These are *starting hypotheses* whose packaging may not work, not prescribed final dimensions. Ensure standing height calculation includes pelvis and flexion. Keep motor volume inside a bounded, narrow pelvis/thigh/knee envelope where possible; if a motor won't fit, show it in the visual rendering and flag the violation. Mass distribution must include servos, brackets, shafts, fasteners, wiring, wiring guards, electronics placeholders, and realistic material densities.

Define reference frames clearly: world Z up, hip axes and foot local axes explicit. No interpenetrating solids. Center of mass, joint inertias, and contact geom orientations must be inspectable.

## 4. Physically purchasable actuation constraint

**No ideal unlimited torque motors.** Select real, identifiable off-the-shelf smart servos (serial bus / encoder feedback preferred), with sources, official specifications, approximate cost, voltage, dimensions, mass, stall torque, rated/continuous behavior when available, command frequency, thermal and speed limits. Record exact model and voltage for every joint. Possible candidate families to evaluate:

- **Waveshare SC15**: 4.8–8.4 V bus servo, quoted up to 17 kg·cm at 8.4 V, approximately 40.5 × 40 × 20 mm; potentially bulky relative to our leg diameter. Source: https://www.waveshare.com/wiki/SC15_Servo
- **ROBOTIS DYNAMIXEL XL330-M288-T / M077-T**: small, comparatively inexpensive servos, vendor lists 0.52 Nm (M288) or 0.22 Nm (M077), at 5 V; verify which torque spec is stall, speed, mass, and whether duty cycle suffices. Official product catalog: https://en.robotis.com/shop_en/category.php?ca_id=20
- A higher-torque commercially available smart servo only if justified by an explicit mechanical feasibility and cost check.

The above are **candidate references, not preapproved adequate actuators**. A lower-cost hobby servo advertised as “30 kg” generally quotes a stall-torque equivalent in kg·cm, not a continuous torque rating. Do not confuse either with payload. Compare all data in **N·m**. For example, 17 kgf·cm ≈ 1.67 N·m *advertised peak/stall-equivalent*, not safe continuous output.

**Required actuator feasibility report:**

1. Link inertial properties and each joint's maximum gravity torque across standing/squatting poses; dynamic joint torques during gait, including impact/transient peaks.
2. Actual torque vs angular speed in the trajectory, not torque in isolation; if official curves are unavailable, use conservative explicit assumptions and mark uncertainty.
3. Two design checks: permissive **peak torque cap** and deliberately conservative **sustained torque cap**. If rated continuous torque is not documented, treat it as **unknown**, include multiple derating scenarios, and do not claim long-term thermal feasibility.
4. Predicted motor temperature/energy only if credible motor thermal/electrical data exists; otherwise report an unresolved verification task.
5. Motor envelope, total servo bill, voltage/current supply budget, control interface, communication/latency constraints.
6. Compare joint *simulation* `actuator_force`, velocities, commanded setpoints and acceleration against the selected physical servo specifications, and check saturation over time.

Suggested first-pass **actuator budget: US$250–600**, with a visible fallback if the mechanical layout cannot meet it. This is a planning range, **not** a live quote. Hardware total includes the motors, controller, IMU, power source, wiring, brackets, bearings and prints; track separately. No purchases in this phase.

## 5. Simulation fidelity

Use Python + **MuJoCo** and a maintained RL library (e.g. Stable-Baselines3 PPO for simple prototyping; another library if parallel simulation speed warrants it). Keep control abstraction hardware-transferable:

- Preferred actions: joint position targets or target offsets; apply a **finite-bandwidth, torque-saturated servo controller** at control cadence. An alternate torque-controller baseline is acceptable *for comparison only*, but it is not sufficient evidence that a position-commanded hobby smart-servo robot can reproduce the gait.
- Physics integrator step nominally 0.001–0.002 s; policy/control updates nominally 50–100 Hz. Test stability and benchmark, do not assume the hardware will sustain either update rate.
- PD law / motor dynamics, joint hard stops, friction, backlash/deadband or sensitivity to them, latency, target slew limits, encoder noise, command quantization, and joint losses. Avoid claim of realism if these are omitted.
- Model ground friction and foot collision geometry accurately. No hidden flat collision sole beneath cosmetic stick feet.
- Randomize masses/inertias, motor strength/speed, latency, Coulomb friction, and floor contact friction after learning a deterministic baseline.
- Track actual non-saturated command vs output torque; output logs must show torque and speed violations, not merely clip and conceal them.

MuJoCo reference: https://mujoco.readthedocs.io/en/stable/computation/ . Native actuators can model position control, force limits and transmissions, but *hardware servo behavior should still be calibrated/approximated explicitly*.

**Sensor observations (hardware-plausible):** pelvis orientation/angular velocity from virtual IMU; joint positions and velocities; previous action; command velocity; estimated gravity direction; optionally sparse contact switches if physically installable. In training, use privileged state for reward and diagnostics, not for deployed policy observations. Include observation latency/noise randomization. Do not grant the policy perfect world position or a hidden ground-truth future state during final evaluation.

## 6. Training curriculum

All learned policies start with reproducible baselines, seeded experiments, and no post hoc video cherry-picking.

### Phase 0 — model and engineering sanity checks
- Render static and articulated robot; check forward kinematics, collision geometry, joint signs/limits, contact impulses and inertial parameters.
- Estimate gravity/peak torque in multiple poses; manually drive servos from safe crouch to near upright.
- Establish a simple PD stand-pose controller, simulation viewer, and deterministic tests. If even static gravity holding demands unrealistic torque, **redesign geometry/scale before RL**.

### Phase 1 — assisted stand-up / stable pose
- Begin with favorable upright resets or an explicitly temporary stabilizing support/curriculum.
- Train to keep pelvis above target height and pitch/roll near vertical while minimizing joint effort and slip.
- Remove *all* stabilization for the evaluation. Test quiet stand for 5/15/30 seconds, record number of corrective steps and base drift.
- Critically, **true point feet may not permit robust motionless standing without constant control/micro-stepping**. A negative result is valuable: distinguish truly stationary stance from dynamic tiny steps.

### Phase 2 — stepping and forward walking
- Add commanded forward velocity target (initially modest, e.g. 0.05–0.20 m/s scaled for robot size). Allow recovery stepping rather than demand stillness.
- Reward velocity tracking, stable pelvis height/orientation, forward progress, alternate/usable contacts; penalize falls, excessive impacts, motor saturation, energy use, unnecessary swinging and foot slipping.
- Do not over-reward survival so strongly that policy learns to freeze; do not over-reward distance so strongly that it learns to fall forward.
- Train starts and stops; use directional velocity commands only after reliable forward locomotion.

### Phase 3 — robustness and sim-to-real stress
- Test untrained seeds, masses, friction, slope, sensor noise, target latency and restricted actuator strength.
- Add light disturbances/pushes with known magnitudes and directions, record recovery without external aids.
- Evaluate point-tip and rocker-tip robots on **the same held-out random seeds and perturbations**, with matched experiment budgets.

**Optional:** privileged critic/asymmetric actor-critic, imitation from hand-designed trajectories, or curriculum pose shaping if PPO struggles. Clearly distinguish these from purely model-free learning.

## 7. Evaluation: strict, reproducible acceptance gates

Before reporting results, generate evaluation scripts with deterministic seeds and a manifest of training/evaluation configurations. Suggested initial targets (not guaranteed):

| Test | Initial target | Evidence |
|---|---|---|
| Geometry | Matches silhouette targets; motors are physically packaged | front/side views, dimensions, BOM |
| Free stand | >=15 s in >=70/100 unseen resets | complete test video + survival histogram |
| Walk forward | >=2 m in >=70/100 trials without falling | commanded vs actual speed, videos |
| Starts/stops | complete forward→stop→forward command sequence in >=50/100 trials | velocity and pose plots |
| Hardware plausibility | all accepted trials obey explicit servo caps | torque-speed logs and violation counts |
| A/B comparison | both foot geometries tested with matched budget | stability, success, actuator load, slip |

If 2 m is excessive for initial progress, use staged milestones (0.25 m, 0.5 m, 1 m, 2 m) and label them honestly. Always report median, variability, and failure modes. Additional metrics: fall rate, stationary step count, energy proxy, CoM/base oscillation, peak/sustained joint torque ratio, joint speed utilization, number of actuator limit events, and contact impulse. Evaluate with and without domain randomization. Include a baseline policy and any simple heuristic controller attempted.

**Failure taxonomy:** inadequate torque; packaging collision; insufficient balance authority; control-frequency/latency; numerical contact instability; reward exploitation; contact slip; inability to stop; fall during transitions; brittle policy under randomized physics.

## 8. Design studies and mandatory ablations

1. **Foot geometry:** A point-like rounded tip vs B short, narrow rocker; vary contact width and rocker curvature.
2. **Ankle actuation:** 10-DoF with ankle pitch vs 8-DoF without it. Consider mass location and cosmetic bulk.
3. **Scale:** 25 cm / 32 cm / 40 cm option only if physical motor size, torque, and mass budgets are recomputed for each; scaling mass without resizing servo envelopes is invalid.
4. **Actuator strength:** nominal vs conservative sustained torque envelope. Explicitly report which results rely on optimistic motor characteristics.
5. **Noisy sensors / delay:** ideal observations vs measured/realistic conditions.
6. **Waist width:** show how narrowing hips affects collision clearance, balance and motor placement.

Where compute limits prevent all combinations, prioritize A vs B and realistic vs optimistic actuator constraints; document skipped studies.

## 9. Proposed code/repository layout

```text
slender-biped/
  README.md
  docs/
    project_spec.md
    morphology.md
    actuator_research.md
    actuator_feasibility.md
    sim2real_risks.md
    experiments.md
  assets/
    renders/
    videos/
  configs/
    geometry.yaml
    actuators.yaml
    train_stand.yaml
    train_walk.yaml
    eval.yaml
  models/
    slender_biped.xml           # MJCF, or generated model
    generate_model.py          # optional parameterized generator
  slender_biped/
    env.py
    actuator.py
    reward.py
    observations.py
    randomization.py
    evaluation.py
    metrics.py
  scripts/
    inspect_model.py
    torque_audit.py
    train_stand.py
    train_walk.py
    evaluate.py
    render_rollout.py
    compare_feet.py
  tests/
    test_model_integrity.py
    test_actuator_limits.py
    test_contacts.py
    test_reproducibility.py
  results/                    # small summaries/plots; large checkpoints ignored
```

README includes quickstart commands, reproducible environment dependencies, 3D model views, before/after videos, exact configuration of the best policies, clear experiment table, and failures/limitations. Save source models and training code, not just an animation. If CAD/printing is implemented in phase 2, add parametric CAD files and exploded assembly views; for phase 1, CAD-grade printable engineering is desirable but not a blocker for training.

## 10. Workflow and decision gates

**Gate A — Design first:** generate 2–3 geometry sketches/parametric variations, motor envelopes, initial COM, two foot alternatives, dimensional drawings. Select best aesthetics within actuated feasibility. No training yet.

**Gate B — Torque audit:** compute gravity torque and motor speed envelope, verify joints under realistic commands. If clearly infeasible, reduce height/mass, choose compact alternative motors or revise topology. Do not proceed with an implausible model and bury the issue.

**Gate C — Functional simulation:** validate free-base dynamics, contact geometry, finite torque, and servo control. Produce an interactive viewer and random-action/no-actuation sanity clips.

**Gate D — Standing training:** train and test, report quantitative successes/failures; iterate morphology and rewards while retaining experiment versions.

**Gate E — Walking training:** evaluate forward motion, start/stop and limits; compare A vs B using fixed held-out seeds.

**Gate F — Portfolio:** release reproducible model, code, rendered design, charts, concise engineering analysis, and next-step hardware BOM. Distinguish *simulated feasibility* from *validated real hardware capability*.

## 11. Physical replica pathway (later, not necessary for POC)

Design from the beginning for common printing materials, off-the-shelf bearings, standard M2/M3 hardware, common bus servos, accessible assembly screws, replaceable rubber foot caps, wire strain relief, and removable covers. Include an IMU, motor bus interface, power distribution and a safe off switch in the hardware proposal. Physical experimentation should begin **on a tether or low overhead harness** over a padded surface, with torque/temperature monitoring and emergency shutdown. Low-profile tip friction and floor quality will strongly influence outcome. For a physical replica, revisit continuous torque and heating with bench tests **before** attempting untethered walking.

## 12. Definition of done

This POC is successful if it produces:

- An aesthetically faithful, sufficiently slender biped lower-body model with **literal point/stick feet** or nearly invisible curved micro-feet;
- A mechanically reasoned actuator map naming motors a hobbyist could actually purchase, not magical ideal joints;
- Independently tested policies for **standing and walking** in a free-base physics simulation, with complete reported success rates (or rigorously documented negative findings);
- A controlled quantitative comparison of the two foot concepts;
- A credible, costed next step toward printing and building the same geometry at small scale;
- A polished, convincing public robotics-design / learning-control portfolio artifact.

**Success is not defined as making an impressive video at any cost.** It means determining whether, *under explicit affordable-motor constraints*, the form factor can achieve the desired behavior—and finding out exactly which mechanical and control tradeoffs matter.

## References / starting points

- MuJoCo actuation: https://mujoco.readthedocs.io/en/stable/computation/
- MuJoCo source: https://github.com/google-deepmind/mujoco
- Waveshare SC15 specification: https://www.waveshare.com/wiki/SC15_Servo
- ROBOTIS official actuator catalog: https://en.robotis.com/shop_en/category.php?ca_id=20
- Point-foot research example (Bolt): https://gepetto.github.io/BoltLocomotion/

**Research note:** Manufacturer torque figures (especially stall ratings) do not by themselves guarantee safe continuous torque, speed under load, controller bandwidth, or thermal endurance. All proposed component selections must be checked against data sheets and eventually verified by bench tests.
