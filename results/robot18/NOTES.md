# Round 18: continuous walking, steps that don't need to settle (2026-11-16 11:05-, 8 h GPU)

The goal for the next few rounds: steps so quiet that the robot can take the next one right away, without waiting
for the body to stop shaking. The robot should step continuously, as humans and high-end humanoids (Unitree G1) do,
instead of step, stop, step.

## Starting point: l9_s18 on the robot ([robot17/NOTES.md](../robot17/NOTES.md))

The softest steps so far, but four things work against continuous walking.
1. **The swing ends skimming the rug** (the "trip"). The last third keeps a flat board within ~10 mm of the floor.
   Its equal hip + knee lift moves the foot on a circle about the hip, so the descent is a backward sweep along the
   floor.
2. **The hip-roll targets jump 4.4 deg** at every step boundary: the release after touchdown runs past the end of the
   step.
3. **The gait is a sequence of stops.**
   - The weight shifts over (ease in/out), holds, releases to the centre and stops there.
   - Pelvis forward motion happens only during the swing.
   - The settle gate stops the clock for 8 % of the walking time (simulator).
4. **The servos lag.** The swing foot trails its command by ~0.1 s on the robot. c_S9mid18 asks 70 rad/s^2 of the
   knees and 242 rad/s^2 of the hip rolls (the jump) from servos that ramp at 25-60.

## What humans and G1 do (searched 2026-11-16)

- **Human walking:** the centre of mass moves on one smooth side-to-side sinusoid per stride (~6 cm), peaking late
  in mid-stance. Double support is ~20 % of the cycle. Nothing stops. [AAPM&R](https://now.aapmr.org/biomechanics-normal-gait/);
  [Kinematics of gait](https://ouhsc.edu/bserdac/dthompso/web/gait/knmatics/gait.htm)
- **G1 RL locomotion** (unitree_rl_gym, Peng et al.
  [2505.20619](https://arxiv.org/html/2505.20619v1)): a fixed-period gait clock, no waits between steps.
- **Robot gait planning:** ~20 % double support per step
  ([patent 5808433](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5808433)). Zero foot velocity and
  acceleration at touchdown (Honda, round 17).
- **This robot:** at its size, a human-like cadence (Froude-scaled ~3.8 steps/s) is far beyond its servos. It stays
  quasi-static (~1 step/s), but continuous.

## The new reference: `shift_gait.py walk`

Built for continuity: every joint target is smooth and periodic, with no settle, no hold and no jump.

- **Sway:** one periodic function per cycle, a sinusoid flattened at the turning points with a 3rd harmonic (`--flat
  1`: the 2nd derivative is 0 there) or 3rd + 5th (`--flat 2`). The body is ~88 % of the way over the stance foot at
  lift-off and touchdown, 100 % at mid-swing, and crosses the centre at speed.
- **Fore-aft:** the pelvis advances at a constant speed; each stance foot moves back at v for its whole stance.
- **Swing:** the board centre follows a planned path through the leg IK.
  - Up over 45 % of the swing (quintic: zero speed and acceleration at both ends).
  - Forward by 80 % of the swing (zero speed at both ends).
  - Straight down over the last 45 % (quintic: lands with zero speed and acceleration).
  - The board tips toe-up in the air (the leading edge high) and is level at touchdown.
  - No forward motion near the floor.
- **Joint accelerations:** knee 54-78 rad/s^2, near the servos' ramp (c_S9mid18: 70, plus the 242 jump). The largest
  change per phase step is 0.04 rad (c_S9mid18: 0.077, the jump).

| table | step | swing | double support | centre lift | sway | lowest board point over the swing (mm, 0.01 m/s row) |
|---|---|---|---|---|---|---|
| c_S9mid18 (l9) | 1.2 s | 0.6 s | 0.6 s | bump | ease, hold, release | 0 2 9 24 38 44 36 20 9 6 4 2 0 |
| c_W1 | 1.2 s | 0.6 s | 0.6 s | 40 mm | flat 1 | -1 -1 1 8 14 18 19 20 18 13 7 3 -1 |
| c_W2 | 1.0 s | 0.55 s | 0.45 s | 40 mm | flat 1 | ~ c_W1 |
| c_W3 | 1.2 s | 0.7 s | 0.5 s | 45 mm | flat 1 | ... 21 22 23 21 13 6 3 -1 |
| c_W4 | 1.0 s | 0.6 s | 0.4 s | 40 mm | flat 2 | ~ c_W1 |

The lowest point mid-swing is the heel (the board is toe-up), ~20 mm. The toe is ~60 mm up.

Zero-shot, l9's policy on c_W1 / c_W2 (simulator): stays up 84 / 78 %, but walks only 10-13 cm and lands at 270-280
mm/s. It needs training on the new table.

## Runs (20M each from l9_s18 20M on its recipe, one lane at a time; gait_hz and the gate lift-offs follow the table)

[robot18_common.sh](../../scripts/robot18_common.sh), [robot18_queue.sh](../../scripts/robot18_queue.sh)

| run | table |
|---|---|
| n1_w12 | c_W1 (l9's cadence: isolates the new shape) |
| n2_w10 | c_W2 (1 step/s) |
| n3_w12l | c_W3 (longer swing, shorter double support) |
| n4_w10f | c_W4 (1 step/s, flatter sway) |

Evaluation per lane: `gait_eval` robot (`--profile`: swing height by the clock, real touchdown vs scheduled, gate
holding %) and loose, vx 0.03, push_eval walk robot + loose.
