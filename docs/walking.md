# Walking on the big pads: what we tried and what we learned (2026-11-08 / 09)

The `F_board150w` snowboard pads ([foot_pads.md](foot_pads.md)) made standing on a rug nearly passive. Walking turned
out to be a different problem. This document records every approach we tried, in simulation and on the robot, why it
did or didn't work, and what we learned about this frame. The RL walker round in detail is in
[walking_b150.md](walking_b150.md).

**Where we are:** no approach walks reliably on the robot yet. The best robot result is a single deterministic step
with a hip-roll weight shift that "sometimes works". In simulation, several walkers walk; on the robot they fall on
the first or second step. The reason is structural (§1), and it shapes everything that follows. In simulation, an
ankle roll servo per leg removes it: one-foot balance for seconds, and six steps without falling in nearly every
hard-floor walk (§8, §9).

## 1. Why this frame is hard to walk with

Numbers for the legs-only robot as tested (0.95 kg, `hw_pad_board150w_legs_8dof`):

| quantity | value |
|---|---|
| centre of mass height | 258 mm |
| boards | 120 × 50 mm contact (150 mm with the upturned ends), centres at y = ±40.5 mm |
| inner / outer board edges | ±15.5 mm / ±65.5 mm |
| leg | 4 joints: hip yaw, hip roll, hip pitch, knee. No ankle |
| thigh / knee-to-board | 138 mm / 144 mm |
| servo motion profile | acceleration ≤ ~35 rad/s² (knee, hip pitch), ~7 rad/s² hip roll in the fitted model; ~0.1 s lag |

**Single support has no margin.** With both boards down, the centre of mass sits on the midline, 15.5 mm *outside*
the inner edge of either board. Without an ankle roll joint a flat board fixes its shin's lateral tilt, so the robot
can't move its weight over one board. Lifting a foot starts a sideways fall toward the lifted foot.

**The fall is fast compared with the servos.** As an inverted pendulum the tilt grows by a factor e every
√(h/g) = 0.16 s. A swing takes ≥ 0.3–0.4 s on the real servos (acceleration limit plus ~0.1 s lag), so the initial
offset grows about tenfold before the foot is down. Measured on the robot: 2–5° of roll toward the lifted foot by
touchdown, then a roll rate of 20–40°/s that a rigid, flat board can't absorb.

**A flat board fixes the shin angle.** Each leg has only two sagittal joints (hip pitch, knee). With the board flat,
the shin angle is fixed, so:

- the foot can't be lifted without large knee travel: lifting a level board from a nearly straight rear leg needs
  ~40° of knee in a swing, far beyond what a 0.3 s swing allows at 35 rad/s² (~10°);
- two flat boards can only be staggered fore-aft in one configuration: front knee bent, rear leg nearly straight,
  thighs symmetric about vertical, which puts the pelvis ~38 mm ahead of the standing pose (§4);
- skiing (alternating glide with both boards down) is impossible: staggering two flat boards tilts them (§3).

**Standing is the opposite.** On two boards the support spans 131 mm side to side, which is why the robot is so hard
to tip while standing.

## 2. RL walkers on a level-sole reference (round 1)

Residual PPO policies on a reference gait (2.4 steps/s, 33 % swing), trained on simulated rugs.
Detail: [walking_b150.md](walking_b150.md) §1–5.

- In simulation: `b150_walkstride_L` 80–100 % survival over 20 s, 3 cm/s; `b150_walksafe_L` 90 %; full body
  `b150_walksafe_F2` 70–80 %, mostly stepping in place.
- **On the robot (~30 trials, 42 logs): first-step falls.** The safe walker fell on the first step almost every time;
  the walker managed 3–4 steps 5 % of the time; ~70 % of falls happened on the first step, 15 of 20 early falls to the
  left (toward the first lifted foot).
- Ruled out in simulation: the start-up ramp, starting lean, IMU offset (≤ 5°), mass, servo gains and delay. Only a
  **softer rug** reproduced the robot's falls. Our rug model had scaled the contact softness by pad area, which made
  the training rugs nearly firm for the big boards (the board sank ~1 mm on the "deep" rug); the real rug is softer.
  Lesson: check that the simulated floor reproduces the robot's failure before training more.

## 3. Things that can't work on this frame

- **Skiing** (five runs): every policy stood still. Two flat boards side by side can't slide alternately without
  tilting (§1).
- **A toy-walker rock** (`slender_biped/rock_walk.py` `RockWalker`, `scripts/rock_opt.py`): rock the robot onto the
  outer edge of one board so the other lifts. The robot has a ~1.2 Hz side-to-side resonance, driven well by
  opposite hip rolls (±0.1 rad → ±10° rock, ~9 mm lift). On hard floors the rock grows until it tips over in the
  first second; on rugs the soft contact absorbs it and nothing lifts. An evolution-strategy search over the
  controller's 13 parameters converged to standing still.

## 4. Step-and-settle RL walkers

Request: small quick steps with a pause after each, so rocking can't build up; the first step is the hard one.

- **Reference** `b150L_settle` (`ref_gait.py --joint-swing`): 0.3 s swing, then both boards flat and nothing moving.
  Staggered flat boards forced the "ready" pose (pelvis 38 mm forward, centre of mass 31 mm ahead of the board
  centres). The forward command ramps up from zero (first steps millimetres long); episodes start in the ready pose.
- **Reward traps.** Two runs stood still. The velocity-tracking kernel pays ~half its maximum for standing at a
  4 cm/s command, and the contact/tip-height imitation terms pay for both boards resting on the floor. What worked:
  `progress` (distance per gait cycle as a fraction of the command; zero for standing) and `ref_swing` (the scheduled
  foot is actually up; zero for standing), with the imitation terms off.
- **Simulation:** `b150_step3_L` cut first-step falls on the soft rug from 20/20 (old walkers) to 2/20, but slid
  60–80 % of the time and fell forward.
- **On the robot:** stuck in place, the boards' toes digging into the rug (the centre of mass was too far forward).
- **Slower, leaned back, with a settle gate** (`b150_step4_L`): one step per second, the pelvis leaned back 15° (the
  one free angle with staggered flat boards: centre of mass 32 → 21 mm ahead), and the gait clock waits at each
  lift-off until the gyro reads < 0.3 rad/s. Simulation: 0–1 first-step falls in 20; mostly sliding on rugs. On the
  robot: a "weird" stance, stuck in place until it tripped itself.

Lesson: the reference gait shapes the walking stance; a pose that only exists to satisfy a kinematic constraint
(flat staggered boards) looked wrong and behaved badly on the real rug.

## 5. Deterministic single steps

`slender_biped/rock_walk.py` `QuickStep`, `scripts/det_walk.py` (simulation), `policy_stand.py run --run step`
(robot). From the default stance: wait, swing the right leg (hip-pitch and knee lift bump, board lands `fwd` ahead,
heel first), hold.

- **Forward only fails.** While the foot is up the robot rolls toward it; it lands rolled 6–8° on the board's outer
  edge and pivots over it. Simulation: a 5 cm step fell on all floors.
- **A little outward (`out` = 3–6° hip roll) fixes it in simulation** (5 cm step, 4–5 of 5 seeds up on every floor).
  **Not on the robot:** the frame was past recovery before the step finished. Logs: the knee peaked 0.1 s later than
  planned (servo lag), roll −2 to −5° at +0.3 s, then −10 to −15° at +0.6 s.

## 6. Balancing on one foot with hip roll

- **No static one-foot pose exists within the joint limits.** An optimizer over all 8 joints (stance board flat,
  other board lifted ≤ 30 mm, hip yaw ≤ 10°) puts the centre of mass at best −0.3 mm from the standing board's edge
  (outside), with both hip rolls at their −14° adduction stop. More range barely helps: +3 mm at −20°, +7 mm at −30°.
  A static one-foot stance on this frame needs ~33° of pelvis roll.
- **The parallelogram weight shift.** Rolling the stance hip inward and the swing hip outward by the same angle slides
  the pelvis sideways, level, with both legs leaning and both boards rolled onto an edge. At ~0.16 rad the centre of
  mass is over the stance board's centre; beyond ~0.24 rad it passes the outer edge and topples. But the other board
  still carries weight (the robot rests on two edge lines), so lifting it tips the robot back. Single support is a
  knife edge.
- **Shift, then step.** Shifting before the step shrinks the initial fall. Simulation: roll toward the swing foot at
  +0.3 s drops from −5…−6° to −1.5…−3.4° with shift 0.15 rad released during the swing.
- **On the robot (27 trials, shift 5.7–8.6°):** 5.7–6.9° of shift leaves the robot leaning 2–3° toward the stance
  side; 8.6° over-shifts (+9.5° before the step). **Through the swing the shift and the step cancel** (roll stays
  0 to +4°). The falls come *after touchdown*: the released shift unwinds and the robot swings to −5…−9°; about half
  recover, the rest go on to −16…−20°. `--det shift=0.10 release=1 fwd=0.03` "sometimes works".

## 7. Shift-and-step reference (2026-11-09)

From the robot logs of §6: the shift and the step cancel during the swing, and the falls come after touchdown. So
the next reference gait (`slender_biped/shift_gait.py`, table `b150L_shift`) builds both into every step, slowly,
with a full rebalance in between. Per 2.5 s step: settle 0.4 s; hip-roll shift 0.8 s (stance leg inward, swing leg
outward, 0.10 rad); 0.35 s swing, the shift blended out over the swing and another 0.4 s; settle. Legs alternate. The
staggered boards keep their centres at the stance height (they tilt ~±2° for a 3 cm stagger); placing the lowest edge
on the floor instead shortened both legs and sat the robot back on two board edges (falls in 2 s).

- **Played open loop, with no policy** (`scripts/det_walk.py shiftwalk`, robot: `--run shiftwalk`): 16 of 16
  simulated walks stay up for 16–20 s, with 7–8 steps on a hard floor (5–9 mm lift) and fewer, barely lifted, on
  rugs. It doesn't go anywhere (−11 to +3 cm).
- **RL on top** (task `shift_b150`): residual policy for forward progress and lift; the settle gate holds the clock
  at each lift-off until the body turns slower than 0.3 rad/s (up to 1 s). Results: §9.

## 8. Ankle roll: a design study

Adding an ankle roll joint per leg (scripts/make_ankle_model.py: hinge 30 mm above the board, along it, driven by
the hip-roll servo STS3215 C001 1:345 at 7.4 V, 70 g servo + bracket on the shin; model
`hw_pad_board150w_legs_ankle_10dof`, 1.11 kg). Deterministic controllers in `scripts/ankle_study.py`.

- **The weight shift becomes a real shift.** Leaning both legs by φ with the hip rolls while the ankle rolls keep the
  boards flat moves the pelvis sideways, level, with both boards flat on the floor. φ = 0.22 rad (12.6°, inside the
  −14° hip-roll stop) puts the centre of mass on the stance board's centre.
- **Standing on one foot works.** On a hard floor and the firm rug, the robot shifts over the left board, lifts the
  right board 10–19 mm and holds it for 3 s with the centre of mass within 0.3 mm of the board centre and 1.3° of
  tilt, then puts it down and shifts back (6/6 seeds), when the foot is put down with 0.1 rad of knee bend left in
  (the pelvis droops 1.4° on the lifted side, so a fully extended leg lands early and pushes the robot over the
  stance board). Without ankle roll, no static one-foot pose exists within the joint limits (§6).
- **On soft rugs the open-loop shift overshoots**: the loaded board sinks on one side and rolls, and the robot
  topples during the shift.
- **Six open-loop steps didn't work.** The first step is clean (lift 10 mm, 1.4 s on one foot, landing); the second
  fails in the weight transfer: whatever the landing leg does with its residual knee bend (keep it: the robot leans
  onto the short leg; release it fast: it pushes the robot over the stance board), the transfer goes wrong. Neither
  IMU feedback on the ankles nor on the stance hip, nor a centre-of-mass correction of the shift, fixed it. This is
  where closed-loop control (or RL) is needed; RL on the ankle model: §9.

## 9. RL on the shift-and-step reference, and one-foot balance (2026-11-09)

Goal: six steps without falling. Test (`scripts/count_steps.py`): a step is a board lifted > 3 mm for ≥ 0.1 s that
lands ≥ 1 cm further forward; 20 s walks from standing at 1.2 cm/s, the robot's IMU filter in the loop, randomized
servos and masses. One-foot balance (`scripts/balance_eval.py`): longest time with the right board up and the left
down.

| run | what changed | hard floor: 6+ steps / falls | firm rug | medium rug |
|---|---|---|---|---|
| `b150_shift_L` 10 M | residual on `b150L_shift`, settle gate, progress + stride + lift rewards | 3/8, 0 falls (median 2) | 0/8, 0 falls | 0/8 |
| `b150_shift_L` 20 M | same, trained on | 0/8 (median 0) | 0/8 | 0/8 |
| **`b150_shift2_L` 10 M** | no progress reward, stride 60, lift 15, slip −300 | **3/12, 5 falls (median 3, best 10)** | 1/12, 3 falls | 0/12 |
| `b150_shift2_L` 20 M | same, trained on | 0/8, 8 falls | 0/8, 8 falls | 0/8, 8 falls |
| `b150_shift3_L` 10 M | + IMU filter and a ±3° tilt-reading offset in training | 1/10, 10 falls | 1/10, 8 falls | 0/10 |
| `b150_shift4_L` 10 / 20 M | from `b150_shift2_L` 10 M, falling −100, stride 40, lift 10 | 2/12, 11 falls → 1/12, 12 falls | 3/12, 10 falls → 1/12, 12 falls | 0/12 |
| **ankle roll `b150A_shift3_L` 10 M** | 10-joint model, gentler shift reference `b150A_shift`, stride 20, lift 6, slip −50 | **16/16, 0 falls (median 9)** | — | **14/16, 2 falls (median 8)** |
| ankle roll `b150A_shift3_L` 30 M | same, trained on | 10/12, 1 fall | — | 7/12, 1 fall (soft rug: 3/12, 2 falls) |
| ankle roll `b150A_shift3s_L` 10 M | from `b150A_shift3_L` 10 M on softer rugs (board sinks up to ~12 mm) | 12/12, 0 falls | — | 10/12, 0 falls (soft rug: 1/12, 0 falls) |
| ankle roll, torso and head `b150A_shift_F` 10 M | full-body ankle model, from `b150A_shift3_L` 10 M | 10/10, 0 falls | — | 5/10, 0 falls (soft rug: 1/10, 5 falls) |
| ankle roll, torso and head `b150A_shift_F` 30 M | same, trained on | 9/10, 0 falls | — | 5/10, 0 falls (soft rug: 3/10, 4 falls) |
| **ankle roll `b150A_shift3i_L` 10 M** | from `b150A_shift3_L` 10 M with the IMU filter and a ±3° tilt-reading offset | **10/10, 1 fall** | — | **10/10, 0 falls** |
| ankle roll `b150A_shift3i_L` 20 M, +2° roll offset | same, tested with the robot's apparent offset | 10/10, 0 falls | — | 5/10, 0 falls |

- **Stepping and staying up trade off sharply.** With progress rewarded, the policy slides instead of stepping (the
  slip penalty, on velocity squared, barely notices 1 cm/s). With steps rewarded, it steps more and falls more as
  training goes on; the best checkpoints are early (10 M). A slip penalty of −2000 dominated everything, so falling
  early became the cheapest way out.
- **Training with realistic sensing (IMU filter, tilt offset) made falls more common**, consistent with the
  filter's lag mattering more for stepping than for standing.
- **Balance on one foot, as built** (`b150_bal_L`, 30 M): never falls, but holds the foot up 0.05–0.18 s (median),
  0.44 s at best: brief hops, not balance. This matches §6.
- **Making falls expensive doesn't help the frame as built** (`b150_shift4_L`, termination −100): it fell more as
  training went on (35 of 36 walks at 20 M). No reward setting we tried gave reliable stepping on this frame.
- **Balance on one foot with ankle roll** (`b150A_bal_L`, 30 M): **1.3–2.0 s on one foot (median), up to 2.9 s**,
  falling in 1–2 of 6 walks; 10× the frame as built.
- **The ankle-roll walker reaches the goal.** `b150A_shift3_L` (10 M): six steps or more in 16/16 hard-floor walks
  and 14/16 on the medium rug, no falls on the hard floor; steps 5.6 cm, 9 mm clearance, 0.18 s in the air, 1.3 cm/s
  (GIF `assets/gifs/ankle_walk.gif`). The soft rug (board sinks ~6 mm) is still hard: 4/16 at 10 M with 10 falls;
  the 30 M checkpoint falls far less there (2/12) but steps less. The first ankle walker (0.8 s, 0.22 rad shift
  reference, step-forcing rewards) never learned to stay up; the gentler reference (0.16 rad over 1.2 s) and the
  milder rewards made the difference.
- **Softer training rugs make it safer, not better at stepping on them** (`b150A_shift3s_L`): no falls on any rug
  but the softest, but on a rug the board sinks 6 mm or more into, a 9 mm lift is barely a step.
- **Realistic sensing doesn't break it.** Trained with the robot's IMU filter and a ±3° tilt-reading offset
  (`b150A_shift3i_L`), the ankle walker takes six steps or more in 10/10 hard-floor and 10/10 medium-rug walks;
  `b150A_shift3_L`, never trained with an offset, still manages 10/10 and 9/10 with a +2° roll offset. On the frame
  as built the same sensing made falls far more common (§9 table, `b150_shift3_L`).
- **With torso and head on** (1.46 kg with the ankle servos) the ankle walker still takes six steps or more in 9–10
  of 10 hard-floor walks without a fall, and in half the medium-rug walks.

### On the robot (as built)

The robot reads +1.4 to +3.3° of roll standing still on the big pads, before every trial in the logs. A robot on two
flat boards can't hold a lean in simulation, so this is most likely an IMU calibration or joint-zero offset, and every
policy sees it as a lean to the left. Check the pelvis with a spirit level; if it is level, re-run `imu-calib` or add
`--roll-trim 2`. Then, on the frame as built:

```bash
python policy_stand.py run --run b150_shift2_L --params params_0010M.pkl --enable --ramp 0 --vx 0.012   # best stepper
python policy_stand.py run --run shiftwalk --enable --ramp 0 --det vx=0.012                             # the reference alone, no policy
python policy_stand.py run --run step --enable --ramp 0 --det shift=0.10 release=1 fwd=0.03             # one step
```

`b150_shift2_L` takes 6+ steps in about a quarter of simulated hard-floor walks and falls in 40 %; expect less on the
rug. The ankle-roll policies need the 10-joint robot.

### What adding ankle roll would take

- Per leg: one STS3215 C001 (1:345, 7.4 V, the hip-roll servo; 55 g) and a bracket, about 70 g on the shin and
  10 g on the foot, a roll hinge ~30 mm above the board along its length, ±20° of travel (the walker uses ±9°).
- Designed in [hardware/ankle_p1](../hardware/ankle_p1/README.md): the servo's socket on the foot (A02, with the
  board150w sole), the roll fork on a straight shin (A01), the leg straight at rest with the board flat, the leg
  length unchanged. It differs from the study model: hinge 38.55 mm above the sole (study: 32), straight rest pose
  (study: hip 16°, knee 32°), and the servo turns with the foot (~85 g on the foot instead of 10 g). Retrained on it,
  with pads of other shapes and a slow step: [walking_ankle.md](walking_ankle.md) (straight legs don't work; the fork
  is now angled 16° on the straight shin to keep the bent stance).
- Two more bus IDs and a 10-joint runner and policy; nothing else on the electronics side.
- What it buys, in simulation: one-foot balance (2 s and more), and six steps without falling on hard floors and
  firm rugs, with the robot's servo model and IMU filter in the loop. What it doesn't fix: deep rugs, where the
  board sinks most of a step's lift.

## 10. Learnings

1. **Analyse the robot's failure before training more.** Each time, the robot logs pointed to one specific moment
   (first lift, touchdown, after the release), and the simulator only became useful once it reproduced that moment.
2. **The floor model matters more than the policy.** Area-scaled rug softness made the training rugs nearly firm for
   big boards; the robot's falls only appeared on a softer rug.
3. **Reward terms that pay for standing still win.** Velocity kernels, contact and tip-height imitation all pay for a
   robot that doesn't move; use progress and lift terms that are zero when standing.
4. **Servo acceleration, not speed, limits a quick step** (~35 rad/s², ~0.1 s lag). Check reference accelerations
   against the profile, not just speeds.
5. **No ankle roll is the root cause, and ankle roll fixes it in simulation.** Without it every step is a controlled
   fall with ~15 mm of initial offset and a 0.16 s time constant; the hip-roll shift reduces the offset but can't
   remove it, and no reward setting gave reliable stepping. With an ankle roll servo per leg the same shift-and-step
   approach balances on one foot for seconds and takes six steps without falling in nearly every hard-floor walk.
8. **Early checkpoints step, later ones play safe.** In every stepping run the step count peaked around 10 M and
   training on traded steps for survival (or, with step-forcing rewards, survival for steps). Pick checkpoints on
   the step test, not the reward.
6. **Kinematic constraints of flat boards** (fixed shin angle) explain skiing failing, the ready pose, and the large
   knee travel needed to lift. A reference that ignores them gets rejected by RL; one that obeys them can look odd.
7. **Measure the start separately.** A first-step test (`scripts/walk_start.py`: start from standing, report falls in
   the first seconds and their direction) separated walkers that 20-second survival numbers ranked the same.

## 11. Tools

| tool | what it does |
|---|---|
| `scripts/det_walk.py <lift\|sine\|cpg\|rock\|step\|shiftwalk>` | deterministic controllers in the simulator, all floors, `--view` / `--gif` |
| `policy_stand.py run --run <same names> --det k=v` | the same controllers on the robot |
| `scripts/walk_start.py <run>` | first-step test: 20 starts from standing, falls and their direction |
| `scripts/gait_quality.py <run> --floor-tc --imu` | survival, speed, lift, air time, sliding |
| `slender_biped/ref_gait.py`, `slender_biped/shift_gait.py` | reference gaits (level-sole, step-and-settle, shift-and-step) |
| `policy_stand.py run --params --ready --vx --roll-trim` | robot runner: checkpoint choice, ready-pose blend, speed ramp, settle gate, IMU roll offset |
| `scripts/count_steps.py <run>`, `scripts/balance_eval.py <run>` | six-step test; one-foot balance test |
| `scripts/make_ankle_model.py`, `scripts/ankle_study.py` | 10-joint ankle-roll models; deterministic one-foot balance and walk on them |
| `scripts/shift_queue*.sh` | the 2026-11-09 training queue (one run at a time) |
