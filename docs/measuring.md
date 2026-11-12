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

## From the measurements to training

Yes: with these numbers the simulator can be made to match the robot before any training, and policies trained on it
should behave on the robot as they do in simulation. Each measurement fixes one part of the model rather than adding
a penalty:

| measurement | model parameter | where |
|---|---|---|
| E0 mass, E1 / E3 tip angles | pelvis mass and centre of mass (x, y, height) | `scripts/make_pelvis_model.py` (extra point mass for what is missing) |
| E1 sideways tip / slip, E2, E6 board tilt under load | encoder-invisible give at the hips (k, N·m/rad), randomized around it | `scripts/tilt_test_sim.py` fits it (k ≈ 4, 2026-11-12); `EvalConfig.hidden_flex` and `EnvConfig.hidden_flex` / `dr_flex` (training) |
| E4 strip vs commanded lift | standing-leg sag and give in the vertical, the real clearance | the reference gait's lift, the swing-clearance reward |
| E5 landing tip / roll rate | touchdown stiffness: floor contact softness and damping, servo damping | `floor_tc`, `floor_damp`, `kd` |

Then the gate: `exercises_sim.py predict` must match the robot's E1–E6 within their scatter. Then the robot-like
evaluation of §13–14 (with the measured give) must pass `c3m_T12`, as the robot does, and fail the h policies, as the
robot does. Only then fine-tune from `c3m_T12`, one change at a time.

Real limits do belong in training as constraints: the hip-roll adduction stop (-10°), and the minimum real clearance
once E4 has measured it.
