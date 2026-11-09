# Walking on the big pads: what we tried and what we learned (2026-11-08 / 09)

The `F_board150w` snowboard pads ([foot_pads.md](foot_pads.md)) made standing on a rug nearly passive. Walking turned
out to be a different problem. This document records every approach we tried, in simulation and on the robot, why it
did or didn't work, and what we learned about this frame. The RL walker round in detail is in
[walking_b150.md](walking_b150.md).

**Where we are:** no approach walks reliably on the robot yet. The best robot result is a single deterministic step
with a hip-roll weight shift that "sometimes works". In simulation, several walkers walk; on the robot they fall on
the first or second step. The reason is structural (§1), and it shapes everything that follows.

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

## 7. Learnings

1. **Analyse the robot's failure before training more.** Each time, the robot logs pointed to one specific moment
   (first lift, touchdown, after the release), and the simulator only became useful once it reproduced that moment.
2. **The floor model matters more than the policy.** Area-scaled rug softness made the training rugs nearly firm for
   big boards; the robot's falls only appeared on a softer rug.
3. **Reward terms that pay for standing still win.** Velocity kernels, contact and tip-height imitation all pay for a
   robot that doesn't move; use progress and lift terms that are zero when standing.
4. **Servo acceleration, not speed, limits a quick step** (~35 rad/s², ~0.1 s lag). Check reference accelerations
   against the profile, not just speeds.
5. **No ankle roll is the root cause.** Every step is a controlled fall with ~15 mm of initial offset and a 0.16 s
   time constant. The hip-roll shift reduces the offset but can't remove it.
6. **Kinematic constraints of flat boards** (fixed shin angle) explain skiing failing, the ready pose, and the large
   knee travel needed to lift. A reference that ignores them gets rejected by RL; one that obeys them can look odd.
7. **Measure the start separately.** A first-step test (`scripts/walk_start.py`: start from standing, report falls in
   the first seconds and their direction) separated walkers that 20-second survival numbers ranked the same.

## 8. Tools

| tool | what it does |
|---|---|
| `scripts/det_walk.py <lift\|sine\|cpg\|rock\|step\|shiftwalk>` | deterministic controllers in the simulator, all floors, `--view` / `--gif` |
| `policy_stand.py run --run <same names> --det k=v` | the same controllers on the robot |
| `scripts/walk_start.py <run>` | first-step test: 20 starts from standing, falls and their direction |
| `scripts/gait_quality.py <run> --floor-tc --imu` | survival, speed, lift, air time, sliding |
| `slender_biped/ref_gait.py`, `slender_biped/shift_gait.py` | reference gaits (level-sole, step-and-settle, shift-and-step) |
| `policy_stand.py run --params --ready --vx` | robot runner: checkpoint choice, ready-pose blend, speed ramp, settle gate |
