# Measuring the robot against the simulator

Round 3's policies walked in simulation and fell on the robot, because the simulator was wrong in ways nobody had
measured ([walking_clean.md §14](walking_clean.md#14-why-13s-policies-fell-sideways-the-legs-give-where-the-encoders-cant-see-2026-11-11-evening)).
These exercises measure the few quantities walking depends on, each with the simulator's prediction next to it. Where
they disagree, the model is fixed first, and only then are policies trained on it.

Model: `hw_ank3_b150_legs_pelvis_10dof`, the legs plus the pelvis shell as built (T03 pelvis ring, T01 / T02 hip
panels, 110 g with screws; `scripts/make_pelvis_model.py`). 1.205 kg, centre of mass 247 mm above the soles in the
stance pose (the old legs-only model: 1.095 kg, 230 mm).

## How to run

On the robot (`hardware/leg_p0/software`, the runner's setup; 5 V; a hand next to the robot):

    python exercises.py hold  --enable                                       # E1, E2
    python exercises.py lift  --enable --side R --lifts 10 --hold 60         # E3
    python exercises.py lift  --enable --side R --lifts 2 4 6 8 12 --lower 1.0    # E4
    python exercises.py lift  --enable --side R --lifts 12 --lower 2 1 0.5 0.25    # E5 (separate runs if it tips)
    python exercises.py shift --enable --side L                              # E6

Each writes `ex_<name>_<time>.csv` (the runner's columns plus stage, rep and mark) and a `.json` with the settings.
Keys are logged as marks; Enter or q stops (the robot eases back to the stance pose, torque on). A tilt past 12°
returns to the stance pose fast. Repeat the lift and shift exercises with the other side.

On the PC:

    .venv/bin/python scripts/exercises_sim.py compare hardware/leg_p0/software/ex_lift_R_<time>.csv --flex "" hip_roll=5
    .venv/bin/python scripts/exercises_sim.py predict         # the predictions below

The simulator runs the same scripted joint targets (`exercise_poses.py`, shared) through the evaluator's servo model at
5 V. `--flex` adds encoder-invisible give (N·m/rad; `EvalConfig.hidden_flex`).

Before E1: after re-mounting the IMU on the new slab, run `python policy_stand.py imu-calib` again, against the hip
plate, not with the robot simply standing. It takes roll and pitch separately, each needing the spirit level along one
direction only: plate level sideways, upright then leaning forward (roll zero); plate level fore-aft, upright then
leaning left (pitch zero). `imu-calib --axis roll` (or `pitch`) redoes one and keeps the other. In the model the pelvis
frame is the plate, and the plate is level when both boards are flat; the IMU's zero should mean the same thing. Then
whatever the robot reads standing untouched is real (servo zero offsets, give, build error), and `hold` logs it in its
first 2 s: `compare` reports it and measures the tip angles from it. A standing offset of a degree or more is itself a
finding (the 2026-11-09 stands read +1.4..+3.3 deg of roll): it is what `--roll-trim` / `--pitch-trim` would hide.

## The exercises

### E0. Weigh it

Kitchen scale, the robot as it walks (cables as they hang). Model: **1.205 kg**. More than ~50 g off means parts are
missing from the model (electronics, cables, the IMU board): note what is on the robot.

### E1. Tip-over angles, by hand (centre of mass)

`hold`: the robot holds its stance pose. Tilt it slowly by hand onto the toes of both boards until it stays there by
itself with no force in your hand, and press `f`; the same on the heels (`b`), onto the left board's outer edge
(`l`), and the right's (`r`). Two or three times each.

| direction | model (rigid) |
|---|---|
| forward (toes) | 14.6° |
| back (heels) | 12.8° |
| left / right (outer edges of one board) | 14.9° / 14.9° |

Each angle places the centre of mass relative to the support edges. `compare` turns each difference into how many
millimetres the real centre of mass sits from the model's, using h · tan(angle) with h = 247 mm. 1° at these angles is
about 4.5 mm. Fore-aft says where the mass sits along the boards; left-right asymmetry says it sits off-centre; all four
smaller says it is higher.

**Result (2026-11-12, `ex_hold_20261112_003155` / `_003407`; on the rug).** Standing untouched the IMU reads roll
+0.4, pitch -3.7 deg (`compare --rest 0.4 -3.7`). From there: forward 19.8 deg (15.4 + 3.7 in the first run), back 8.3,
left 5.2, right 5.3.

* Fore-aft: forward + back = 28 deg against the model's 27.4, so the height of the centre of mass is about right
  (~240 mm), but it sits ~20 mm further toward the heels than the model's, mostly because the robot stands leaning
  back 3.7 deg with its boards flat (~15 mm at that height).
* Sideways: ~5 deg against 14.9. This is not the centre of mass (the "43 mm" `compare` prints is meaningless here). The
  model's angle assumes the lifted leg keeps its spread. On the robot the lifted leg swings partly inward as the robot
  tips, which pulls the support in, and once one leg slides beneath the other the fall is certain. The encoders don't
  see it: at the marks both legs read their stance angles (both boards tilted alike, the lifted one ~7 mm up), so the
  swing is the encoder-invisible hip give of walking_clean.md §14.
* **The limit depends on the stance width, and the width ratchets in.** 5 deg is from the default pose. Tilting left
  draws the right leg in; tilting right straight after, without resetting, gives an even lower limit, because the legs
  start closer together. Each side-to-side swing narrows the stance and lowers the next limit. That is how a walk
  that rocks left and right without restoring the width falls sideways within a step or two, and why wider legs help
  so much. The simulator's legs return to their exact spread every time.

**Stance width, measured by hand (2026-11-12, ballpark: it varies with how the boards bend and catch on the rug).**
Model: foot centres 81 mm apart; rigid, that puts the outer edge 65 mm from the centre line and the limit at 14.9 deg.

| start | tilt limit | after letting go | roll at rest after |
|---|---|---|---|
| 85 mm (default) | left +7 deg | 70 mm | +2.2 deg (biased) |
| 70 mm | left +7 deg | 65 mm | |
| 85 mm (default) | right -6.1 deg | 60 mm | -0.7 deg |

* The feet start slipping inward at about +5 deg (left) / -4 deg (right). It is not a fall at once, but from there
  every tilt costs width: 85 -> 70 -> 65 mm, or 85 -> 60 mm in one go to the right.
* A 6-7 deg limit at ~245 mm puts the effective pivot only ~30 mm outside the centre line, near the middle of the
  stance board (which spans 15-65 mm), not its outer edge: the board, the ankle and the hip give before the edge
  takes the weight.
* After narrowing, the robot can rest with a roll bias (+2.2 deg after the left tilts): the give stays loaded,
  locked by the friction of the boards on the rug.

To measure next: the sideways limit from a wider stance (hip roll abducted a few deg).
The simulator needs the give to reproduce the inward swing and the narrowing; policies need to re-spread the legs
each step (they can't see the width directly: it shows only as IMU roll with both feet down).

### E2. Hip give, by hand

`hold`, both boards flat on the floor: push the pelvis gently sideways left and right, pressing `p` while pushing.
Then hold one board down and lift the robot by the pelvis until the other board just leaves the floor (`p`).

Rigid legs: the joint encoders plus the IMU put both boards flat (0°) whatever you do. Any tilt `compare` reports
while you push is give the encoders can't see. In today's walking logs the two boards disagreed by about 5° with both
feet down; the simulator, rigid, by about 2.5°.

### E3. Standing on one foot, and tipping off it

`lift --lifts 10 --hold 60`: the robot shifts 9.5° onto the left board and lifts the right board. While it stands,
tilt it by hand outward over the standing board until it balances by itself (`o`), then inward (`i`).

| | model (rigid, on the left board, right board 10 mm up) |
|---|---|
| outward (over the standing board's outer edge) | 8.0° |
| inward (toward the lifted foot) | 3.6° |
| forward / back | 13.6° / 13.8° |

This is the margin each step has. A robot that tips outward at much less than 8° is the robot of §14. Watch whether
the pelvis droops toward the lifted foot while it stands.

### E4. Minimum foot clearance

`lift --lifts 2 4 6 8 12 --lower 1.0`: one rep per height. The heights are what the joints do with rigid legs. Put a
paper strip under the right board; during each hold, tug it: `c` if it pulls out freely, `n` if it is pinned, `s` if
the board scuffs.

Simulator (rigid; 5 N·m/rad give is similar): the body rolls 1–2° toward the lifted board as the standing leg's servos
sag, so a commanded lift of 2 / 4 / 6 / 8 / 12 mm leaves the board centre only 0 / 0.7 / 1.1 / 2.8 / 6.8 mm up, and
one edge touches the floor up to 8 mm. So the strip may stay pinned well past the commanded height. The gait lifts the
board 15 mm (`c_T12`); the real clearance at a given command is the number the swing reward and the reference gait
need.

### E5. How hard a foot can land

`lift --lifts 12 --lower 2 1 0.5 0.25`: the same one-foot stand, lowered in 2 s down to 0.25 s. Watch each landing;
`compare` reports the peak roll rate. If it tips, the cut-out returns it to the stance pose; catch it, and run the
remaining lowering times separately.

Simulator: **every** landing tips the robot outward over the standing board, even over 2 s (peak roll rate
135–280°/s). The landing leg is commanded back to its full rigid-model length while the standing leg has sagged ~2°.
The landing foot touches early, then the leg keeps extending and levers the pelvis over the standing board: the centre
of mass moves from 13 mm inside the board's centre to past its outer edge. That is the outward tip of §14's policies.
If the robot lands calmly where the simulator tips, the simulator's touchdown is too stiff or its sag too large. If the
robot tips too, the mechanism is real and the policies must land with a shorter leg.

### E6. Weight shift to the hip limit

`shift --side L`: the weight moves onto the left board at 0.5°/s of hip roll, to 9.5° (the bench stops hip-roll
adduction at -10°). Paper strip under the right board: `u` when it pulls free; `t` if it starts to tip.

Simulator: no tip and no unloading within 9.5°; at 9.5° the centre of mass is 8 mm inside the left board's centre. The
encoders plus the IMU keep both boards flat (0°) along the whole ramp. On the robot, `compare` lists the board tilts
per degree of shift: that is the give under a known, growing load, the hip-give stiffness itself.

### E7. Tap test: how the body rings (full body, 2026-11-12 night)

Why: the full-body policies shook at 4.5-5 Hz on the robot, and the simulator shows no shake
([walking_clean.md §18.3](walking_clean.md#183-full-body-steps-but-falls-sideways-and-shakes)).

`hold --seconds 60 --log ex_tap_<dir>.csv`: the robot holds its stance pose. After 2-3 s untouched, give it 3-14 sharp
flicks on the head, about 3 s apart. The log runs at 50 Hz, which is enough for a 5 Hz ring. For each tap, a decaying
sine (frequency, damping ratio) is fitted to the pelvis gyro over the next 0.8 s. The joint motion is the encoders
against the command. Simulator (`hw_ank3_b150_10dof`, k 3.5, pushed at the pelvis under the stance hold): no ring
at all, and the motion dies out within 0.5 s at 0.1-1 Hz.

| tap (log) | what rings | frequency | damping ratio | at the pelvis |
|---|---|---|---|---|
| side of the head (`ex_tap_side`, 3 taps) | roll | 4.8-5.7 Hz | 0.14-0.19 | ±1.0-1.2° |
| front of the head (`ex_tap_fore`, 14) | pitch: a ring plus a slow rock on the boards | ring 4.6-5.6 Hz, rock 0.5-1 Hz | ring 0.13-0.18 | ring ±1°; rock several degrees, onto the heels and over twice |
| back of the head (`ex_tap_back`, 6) | pitch rock; roll ring | rock 0.7-1.0 Hz; roll 5.0-6.0 Hz | rock ~0.35; roll 0.11-0.13 | roll ±0.5° |
| twist (`ex_tap_twist`, 12) | yaw | ~4-5 Hz once released | 0.2-0.4 (back in about 1.5 swings) | ±3-4° |

- **One flexible mode near 5 Hz, damping about 0.15.** It rings in roll and in pitch, and a fore-aft tap rings
  roll too. This is the frequency of the shake in the full-body walks (4.5-5.4 Hz). That makes it a mode of the body
  that the policies drive, not a fault in the policies alone.
- **The joints barely move.** Across all four logs the encoders stay within 0.2-0.6° of the command at the hips and
  ankle rolls. The knees sit 1-1.75° off, and that offset is the steady sag under the body's weight, not the tap. Most
  of the motion is where the encoders can't see it.
- **Twist is stiff and well damped, but not rigid:** the pelvis turns 3-4° on a tap while the hip-yaw encoders move
  at most 0.35°.
- **The full body stands further back.** Pitch at rest in the first second of each log (+ = back):

  | | pitch at rest |
  |---|---|
  | tap tests, full body | +5.8 / +6.3 / +7.5 / +7.2° (side, fore, twist, back) |
  | full-body walks, before the policy moves | +4.1 to +6.0° |
  | legs only: walks, and E1 | +2.5 to +3.7° |

  With the torso and head on, the robot leans 2-4° further back than the legs alone, by an amount that varies by up to
  1.7° between runs. The rigid full-body model tips onto its heels at 9.8°, so a firm tap from the front rocks it
  over.

**By hand** (felt during the taps, not logged):
- From the back, the hip-pitch servo (above the knee) visibly gives, about 1°, and takes part in the rock.
- Twisting the body turns the hip-yaw servos as well.
- The stance hold is the servos' position loop at P gain 32, and it doesn't lock them rigidly.

The encoders see at most 0.4-0.6° at hip pitch and 0.35° at hip yaw, so of the ~1° felt at hip pitch, roughly half
or more happens outside what the encoder sees: in the horn, the bracket or the printed mount around the servo.

**For the model:**
- a 5 Hz, ζ ≈ 0.15 compliance between the upper body and the feet, visible in roll and pitch;
- encoder-invisible give at hip pitch and hip yaw, as well as at hip roll (§E2);
- the full body's backward lean at rest.

**Legs only** (2026-11-13, `ex_tap_legs_{side,fore,back,twist}`, 7-9 taps each on the pelvis ring). By hand the legs
feel much more rigid on their own, and the logs agree:

| | full body | legs only |
|---|---|---|
| roll after a side tap | rings 4.5-5.7 Hz, ζ 0.15-0.19 | **no ring**: a single spike, gone within 0.1 s |
| pitch after a fore / back tap | ring ~5 Hz on top of a 0.5-1 Hz rock | a ~2.5 Hz rock, no 5 Hz ring |
| yaw after a twist | 3-5 Hz, ζ ~0.2-0.3 | 3-5 Hz, ζ ~0.3 (much the same) |
| pitch at rest (+ = back) | +5.8 to +7.5° | +3.8 to +4.7° |

**Hypothesis: the legs and hips are the spring, and the upper body's weight high up turns them into a wobble.**
The logs fit this:
- **The lean is the pelvis's own.** The IMU is on the pelvis, and the pelvis itself leans 2-3° further back with the
  upper body on, so it is the legs (below the pelvis) that give under the extra weight. A torso bending on a
  rigid pelvis wouldn't tilt the IMU.
- **One spring explains both rolls.** In the models, the mass above the hips has 0.8 g·m² of roll inertia legs-only
  (pelvis, centre of mass 23 mm above the hips) and 18.8 g·m² with the upper body (125 mm up): 23 times more.
  - A spring at the hips that rings at 5.2 Hz with the upper body on comes to k ≈ 20 N·m/rad
    (k - mgh = I·(2πf)², mgh = 0.70 N·m/rad).
  - With only the pelvis on top, the same spring and damper would ring near 25 Hz, which the 50 Hz log can't
    resolve. The damping ratio would rise from 0.15 to about 0.7 (ζ grows as 1/√I), so the legs alone barely
    ring. That is what the legs-only log shows.
- **The slow rock** (heel to toe on the boards) drops from ~2.5 Hz legs-only to 0.5-1 Hz with the upper body.
  - The heavier, higher body alone predicts ~1.75 Hz: about 2× the robot's inertia about the floor, and
    m·g·h 4.8 against 3.2 N·m/rad.
  - The rest likely comes from standing nearer the heel edge.
- **Twist is unchanged**: the yaw give is in the legs and boards, and the upper body adds little yaw inertia.

**What the simulator gets wrong.** Its hip give (`hidden_flex`) offsets the encoder by τ/k with a 10 ms lag. That is
a massless, heavily damped give: it can't ring. It is also soft (3.5 N·m/rad per hip, fitted on the large tilts of
E1) where the taps say the small-motion stiffness is about 10 per hip. So the give is probably stiffer near the stance
and softer under large loads (or slack). To fit:
- the give as a real spring and damper with the leg's mass behind it, at hip roll, plus hip pitch and hip yaw
  (felt by hand);
- the fit target: a simulated tap reproducing 5 Hz / ζ 0.15 with the full body, no ring with the legs alone, and E1's
  tilt limits.

### Full body: weight, tilt limits, push and weight shift (2026-11-13)

**E0.** Legs, pelvis and hip plates: **0.9 kg** (model 1.205). Torso and head: **0.3 kg** (model 0.243, lumped
into the pelvis). The model's legs are 300 g too heavy and its upper body too light, so the real robot is
more top-heavy than the model.

**E1, full body** (`ex_hold_full`; the pelvis reads roll +0.3, pitch 4.4° back standing untouched):

| | model, full body (rigid, centre of mass 313 mm up) | robot, full body | robot, legs only |
|---|---|---|---|
| forward | 12.1° | 18.8° | 19.8° |
| back | 9.8° | **7.5°** | 8.3° |
| left / right | 11.9° | **6.2 / ~6.4°**, feet start to slide at 5.8 / 5.7° | 5.2 / 5.3° |

One early `r` mark (at -3.3°) is left out. The sideways limit is the same ~6° as the legs alone: the legs' give and
the feet sliding set it, not the centre of mass. While tilted onto one board, the other leg's knee load drops from
~72 to ~16, so the servo loads show a board unloading without a paper strip.

**E2, full body** (`ex_hold_full_p`): pushing the pelvis slowly sideways with both boards flat rolls it 5.8° one way
and 4.1° the other, while the hip-roll encoders move at most 0.5°. That is several degrees of encoder-invisible give
under a large, slow load, against ±1° of springing at 5 Hz under the taps (E7).

**E6, full body** (`ex_shift_full_L`, onto the left board at 0.5°/s of hip roll). **It tips over the left board after
only 3° of shift** (the simulator: no tip and no unloading through 9.5°):

| hip-roll shift | 0 | 2.0° | 2.7° | 2.9° | 3.0° (stopped, `q`) |
|---|---|---|---|---|---|
| pelvis roll (rest +0.8) | +0.8 | +1.3 | +1.9 | +3.0 | +3.4, then on over to +16.7 while returning |
| right knee load (far board) | 72 | 72 | 72 | 64 | 64 |

- **The pelvis rolls toward the standing board as the weight shifts,** and the roll speeds up: about +0.5° over the
  first 2° of shift, +1.5° over the last 0.3°. That is the give running away under a growing load.
- **The far board was still carrying weight when it went** (knee load hardly dropped). The rigid forward kinematics
  still place the centre of mass 11-15 mm inside the left board's centre at that moment.
- So the full body falls over its own give before it ever stands on one foot. For comparison, the reference gait
  shifts 7.7°.
- In the model the give is a quasi-static τ/k, and it returns the legs to their spread. To get this run right, the
  give needs the upper body's real mass and height on top, a stiffness that drops at large loads, and the narrowing
  of E1.

### E8. Wiggle: how the body answers small commands (full body, 2026-11-13)

`python exercises.py wiggle --enable --pattern <roll|pitch|squat|knees|yaw> [--amp 1.5] [--lean 0]` holds the stance and
adds a sine to one joint pattern at 1-7 Hz (3 s each, 1.5 s still between). It measures, with no policy in the loop,
how far the joints follow a command at each frequency and how fast the body turns per degree commanded: the two halves
of the shake loop (walking_clean §20). `--lean 2` tilts the upper body 2° forward first: the full body stands 6-7.5°
back, at its backward tip limit, and `knees` alone (1.5°) tipped it backward three times in its first second.
`squat` moves knees and hip pitches together so the upper body keeps its angle.

Logs `ex_wiggle_<pattern>_20261113_*.csv`. Per frequency: the joints' amplitude / command (phase in brackets, - = lag),
and the IMU's turn rate per degree commanded on the main axis. Simulator: the round-6 `fitted` settings (without the
knee offset, with which the simulated stance falls backward in 1 s).

| Hz | roll pattern: joints | roll °/s per ° | sim | hip pitch: joints | pitch °/s per ° | sim | hip yaw: joints | sim |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.87 (-38) | 4.8 | 1.4 | 0.71 (-51) | 8.7 | 6.5 | 0.99 | 0.25 |
| 2 | 0.86 (-67) | 6.7 | 5.0 | 0.84 (-60) | 3.4 | 6.2 | 0.92 | 0.24 |
| 3 | 0.72 (-99) | 8.1 | 8.2 | 0.85 (-81) | 8.6 | 4.8 | 0.79 | 0.23 |
| 4 | 0.53 | 8.4 | 7.3 | 0.72 (-113) | **23.0** | 3.2 | 0.58 | 0.21 |
| 4.5 | 0.38 | 7.9 | 6.4 | 0.51 (-127) | **23.8** | 2.6 | 0.43 | 0.19 |
| 5 | 0.27 | 7.6 | 5.1 | 0.42 (-131) | **21.4** | 2.2 | 0.33 | 0.18 |
| 6 | 0.14 | 6.3 | 1.1 | 0.49 (-144) | 16.9 | 1.8 | 0.17 | 0.13 |
| 7 | 0.08 | 5.1 | 0.7 | 0.54 (+178) | 12.1 | 1.4 | 0.10 | 0.09 |

1. **Pitch is the big gap.** Per degree at the hip pitches the real body pitches at 21-24°/s at 4-5 Hz, the simulated
   one at 2-3: 8-10× less. The robot has a pitch resonance at ~4.3 Hz (the fore tap's 5 Hz ring under load) and a
   high response at 1 Hz (the boards' slow rock); the simulator has neither. Per degree the joints actually move, the
   real body turns 47°/s at 4.5 Hz, 1.7× what a rigid body would. Raising the simulated servo gain until its joints
   follow as well as the real ones (kp 32) only brings the simulator to 4.7: the missing part is the body, not the
   servo.
2. **The hip-pitch servo follows half the command at 4.5-7 Hz, ~80 ms late.** The simulated one follows 0.17-0.20.
3. **The roll and yaw servos are acceleration-limited at ~400-500°/s².** Their motion at 4-6 Hz is the same ~0.4-0.6°
   whether 1.5 or 3° is commanded (so the 3° roll run reads half the per-degree response). That matches the older
   hip servos ignoring the acceleration register (their default ramp ~400°/s²); the simulator assumes 25 rad/s²
   (1430°/s²) for them. Roll response is otherwise close to the simulator's up to 5 Hz.
4. **Knees under load don't move for small commands.** `knees --amp 1 --lean 2`: the knee encoders read no motion at
   any frequency (they sit 2.3° behind their command under the body's weight); `squat --amp 1 --lean 2` (knees + hip
   pitches) follows 0.4 at 1 Hz, 0.22 at 5 Hz, and pitches the body 11-13°/s per degree at 4.5-5 Hz. `--lean 2` moved
   the standing pitch only from +5.3 to +4.4°. The shake runs through the hip pitches.
5. Yaw: the real hip yaws follow (0.99 at 1 Hz) but the body barely turns (≤ 1.5°/s per °: the boards stay put); the
   simulated hip yaws hardly move (0.25). Minor.

## From the measurements to training

Yes: with these numbers the simulator can be made to match the robot before any training, and policies trained on it
should behave on the robot as they do in simulation. Each measurement fixes one part of the model rather than adding
a penalty:

| measurement | model parameter | where |
|---|---|---|
| E0 mass, E1 / E3 tip angles | pelvis mass and centre of mass (x, y, height) | `scripts/make_pelvis_model.py` (extra point mass for what is missing) |
| E1 sideways tip / slip, E2, E6 board tilt under load | encoder-invisible give at the hips (k, N·m/rad), randomized around it | `scripts/tilt_test_sim.py` fits it (k ≈ 4, 2026-11-12); `EvalConfig.hidden_flex` and `EnvConfig.hidden_flex` / `dr_flex` (training) |
| E4 strip vs commanded lift | standing-leg sag and give in the vertical, the real clearance | the reference gait's lift, the swing-clearance reward |
| E7 tap ring-down (frequency, damping), lean at rest | compliance between the upper body and the feet (torso joint or springier hip give), encoder-invisible give at hip pitch and hip yaw | model (to do) |
| E5 landing tip / roll rate | touchdown stiffness: floor contact softness and damping, servo damping | `floor_tc`, `floor_damp`, `kd` |

Then the gate: `exercises_sim.py predict` must match the robot's E1–E6 within their scatter. Then the robot-like
evaluation of §13–14 (with the measured give) must pass `c3m_T12`, as the robot does, and fail the h policies, as the
robot does. Only then fine-tune from `c3m_T12`, one change at a time.

Real limits do belong in training as constraints: the hip-roll adduction stop (-10°), and the minimum real clearance
once E4 has measured it.
