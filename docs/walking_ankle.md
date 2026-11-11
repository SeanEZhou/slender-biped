# Walking with the ankle-roll leg as designed (2026-11-09)

The ankle-roll study ([walking.md §8–9](walking.md)) added a roll hinge to the existing leg in simulation. This round
retrains on the leg as it was then designed in CAD ([hardware/ankle_p1](../hardware/ankle_p1/README.md)), and tries
foot pads of other shapes and sizes. The goals: steps that are not "on the verge of tipping" (the real robot is always
worse), a soft step that lifts, swings and lowers the foot while balancing on the other one, and a pad that doesn't get
in the way.

**Next round:** [walking_clean.md](walking_clean.md) (2026-11-10): continuous steps with more lift, trained against
wobble, scuffing and sliding; the policies to try on the robot first.

Everything here is simulation: legs and pelvis (as tested on the robot), the robot's IMU filter in the loop, servo and
mass randomization per walk, floors from hard to a soft rug.

**Summary**

1. **The straight-leg rest pose doesn't work for stepping.** With the leg straight and no ankle pitch, a flat foot
   can't be lifted: a thigh swing doesn't change its height to first order, a knee bend pitches it so the toe digs in and
   shoves the robot back, and a hip hike lifts it only ~4 mm once the servos sag. (§2)
2. **A straight shin with a 16° ankle fork** keeps the usual bent stance (hip 16°, knee 32°) with the roll axis level
   and the board flat. The leg as built in CAD now (`fork_pitch`, §3). The servo in the foot (85–96 g turning with the
   ankle instead of 10 g) still works with the recipe of the ankle study.
3. **Long boards beat small pads.** At 20 M training steps the 150 × 50 mm board takes six steps or more in every walk
   on hard floors and rugs with no falls; the foot-shaped 96 mm pad is next; the 80 × 45 mm rectangle falls in half the
   walks; the 60 × 40 rectangle and the 70 mm disc fall in all. The feet never touched each other in any walk: with
   ankle roll the boards stay flat and apart. (§4)
4. **Quick steps land firmly** (the foot comes down at ~140 mm/s; the body tilts ~7° after each landing, 12–16° at
   worst). A soft-landing penalty didn't change that: the policy follows the reference's quick 0.35 s swing.
5. **Slow steps land softly.** Lift, swing for 1.2 s on one foot, lower over two thirds of the swing: touchdown at
   30–90 mm/s and a third to two thirds of the wobble, with no falls on hard floors and firm rugs (150 mm board). The
   soft rug is still hard for every policy. (§5)
6. **One-foot balance needs the long board, not a light foot**: the study's balance policy holds one foot up 2.2 s on
   this design's heavy foot with the 150 mm board; on the 80 mm pad, trained, 0.4 s. (§6)

## 1. The leg in simulation

`scripts/make_ankle2_model.py` builds models from the design: the straight A01 shin (mass and inertia from the CAD),
the roll hinge 111.45 mm below the knee on the shin centre line, and a foot carrying the servo (55 g), the socket and
rib (14 g), screws and cable (4 g) and the pad, its sole 38.55 mm below the hinge. Visual meshes from the CAD
(`hardware/ankle_p1/cad/build_ankle.py simmesh`). Two variants:

| models | rest pose | ankle axis |
|---|---|---|
| `hw_ank2_<pad>_legs_10dof` | straight (all joints 0), knee range −69° … 110° (it may bend "forward"; the CAD is clear to −90°) | level with the leg straight |
| `hw_ank3_<pad>_legs_10dof` | hip 16°, knee 32° | pitched 16° toe-up on the shin: level in the stance |

Pads (`--list`): `b150` snowboard 150 × 50 (as built), `b100` 100 × 50, `r80` flat 80 × 45, `r60` 60 × 40, `d70` /
`d55` discs, `foot` foot-shaped 96 × 46 (heel 36 mm, forefoot 46 mm). Robot 1.06–1.10 kg.

## 2. Straight legs (the v2 design)

**Reference** (`slender_biped/ankle_gait.py`): feet flat throughout. A flat foot needs a vertical shin (knee = hip
pitch), so a foot behind the hip has the knee bent forward, the thighs pendulum like a parallelogram. Weight shift: both
legs lean (hip rolls), the ankles keep the feet flat; the pelvis centre is over the stance foot at 0.124 rad. Swing-foot
clearance by **hip hike**: the stance leg rolls the pelvis so the swing hip rises by 80 mm × sin ψ; the swing foot then
moves at a constant height, and touchdown is the hike going down, slowly. 4 s per step.

**What happened**

- Open loop (no learning, hard floor, nominal servos) it walks: six steps in 24 s on the 150, 100 and 80 mm pads and the
  foot shape. On a rug and with randomized servos it falls (the weight shift overshoots or drifts: leaning legs are
  neutrally stable and only feedback holds them).
- The foot lifts only ~4 mm of the 9.6 mm the hike asks for (the servos sag under the load); with more hike the robot
  slides sideways with both feet still loaded.
- RL on it: the first reward set (a progress term) shuffled the feet forward without stepping (0–2 steps in 40 s, the
  feet sliding 0.2–0.5 m); weights that paid well for lifting (stride ×60, ref_swing ×15) learned to yank the foot up
  and fall (training episodes got shorter). One-foot balance with a hike-only lift: 0.5 s median.
- Lifting with the knee instead (a flamingo lift, which clears 60–70 mm under the hip by geometry) pitches the foot
  toe-down before it leaves the floor: the toe digs in and pushes the robot over backwards, at any speed.

- A fair retry with the recipe that works for the bent stance (`a2f_b150`: stride ×20, ref_swing ×6, slip −50; 150 mm
  board, 20 M): **never falls** (0/8 on every floor) and wobbles least of all the policies (worst tilt 9–10°), but
  hardly steps: half the hard-floor walks reach six steps in 40 s, the feet slide ~12 cm per step, 5 cm forward in 20 s,
  and on the medium rug it only shuffles.

**Why**: standing straight is a singular pose for lifting a flat foot without ankle pitch. Foot height doesn't change
to first order when the thigh swings with the shin vertical; any knee bend tilts the foot. In a bent stance a little
more hip and knee raises a level foot directly (the old design's lift). The only clean lift from straight legs is the
hip hike, and that is a few millimetres once the servos sag.

## 3. Bent stance with a straight shin (the v3 design)

The CAD keeps the shin straight and pitches only the ankle fork at its lower end by 16° (`fork_pitch`): the roll axis
is level and the board flat in the usual stance (hip 16°, knee 32°), the servo stands upright in the foot, and the
board needs no wedge. Collision free to ±30° of ankle roll (the shin's transition into the fork touches the servo top
at 35°), knee to 110°. In simulation that is `hw_ank3_*`; the ankle study's references (`b150A_shift`,
`b150A_bal1`) and recipe apply unchanged.

The recipe matters: the successful study run used stride ×20, ref_swing ×6, slip −50, no progress term.

## 4. Pads

`shift_b150` task, reference `b150A_shift` (2.5 s per step: weight shift 1.2 s, 0.35 s swing), 20 M steps each, rugs
in training. `scripts/step_quality.py`: 30 s walks at 1.2 cm/s, 6–8 seeds per floor (hard 0.004, firm rug 0.1, medium
0.3, soft 0.8); six steps or more, falls; landing quality.

| pad | hard: 6+ / falls | firm rug | medium rug | soft rug | notes |
|---|---|---|---|---|---|
| **b150** 150 × 50 | **8/8, 0** | **8/8, 0** | **8/8, 0** | 6/8, 0 | +2° IMU roll offset: 8/8, 0 on hard and medium |
| foot 96 × 46 | 6/6, 0 | 6/6, 1 | 0/6, 1 (stuck) | — | 20 M; 20 M more (`a3s_foot2`): never falls but shuffles (0–3/8 with six steps, feet sliding ~0.2 m per step) |
| r80 80 × 45 | 4/6, 3 | 5/6, 1 | 0/6, 2 | — | learns late (10 M: almost every episode falls); 20 M more (`a3s_r80_2`): 6/8, 3 falls; 7/8, 2; 0/8, 2 |
| r80, soft-landing penalty | 4/6, 3 | 5/6, 3 | — | — | landings no softer |
| r60 60 × 40 | 1/6, 6 | 0/6, 6 | — | — | |
| d70 disc | 0/6, 6 | 0/6, 6 | — | — | |

Landing quality, b150: the foot comes down at ~140 mm/s (median; up to ~300), the body tilts 7° within 0.6 s of a
landing (12° at worst) and turns at ~50°/s (up to ~110°/s). Worst tilt over a walk 12–16°; a fall is 40°. The feet
never touched each other (0 of all walks, every pad).

The ankle study's policy (trained on the study model: servo on the shin, 10 g foot), run unchanged on these models:
b150 3/4 walks with six steps, then worse the shorter the pad (b100, r80 1/4; foot, d70, r60 0/4).

## 5. Slow steps: lift, swing while balancing, lower slowly

The ankle study's reference swings the foot in 0.35 s, so every step is a short fall onto the swing foot (the "hard
landing" in its GIF). `b150A_slow1` (`shift_gait.py --hold-shift --rise-frac 0.35`) takes 4 s per step: weight shift
over the stance foot (1.2 s), held through a **1.2 s swing that lifts over 35 % of it and lowers over 65 %**, released
0.8 s after touchdown. The robot stands on one foot for about a second per step; with ankle roll it can.

| policy (b150 pad) | hard: 6+ / falls | firm rug | medium rug | soft rug | touchdown | after a landing: tilt / turn rate |
|---|---|---|---|---|---|---|
| quick step `a3s_b150` 20 M | 8/8, 0 | 8/8, 0 | 8/8, 0 | 6/8, 0 | 139 mm/s | 7° / 53°/s |
| quick step + soft-landing penalty (`a3s_b150_soft`) | 7/8, 1 | 7/8, 1 | 8/8, 0 | 6/8, 2 | 119–156 mm/s | 7° / 46°/s |
| **slow step `a3w_b150` 10 M** | 5/6, 1 | 6/6, 0 | 5/6, 2 | — | **30 mm/s** | **3° / 16°/s** |
| **slow step `a3w_b150` 20 M** | **8/8, 0** | **8/8, 0** | 4/8, 0 | 0/8, 0 (steps in place) | 89 mm/s | 8–12° / 40°/s |
| slow step `a3w_b150` 30 M | 7/8, 0 | 6/8, 0 | 5/8, 0 | 3/8, 3 | 90 mm/s | 6–10° / 37°/s |
| slow step 20 M + landing penalty ×4 (`a3x_b150`, 15 M) | 8/8, 0 | 8/8, 0 | 6/8, 1 | 1/8, 3 | 80–99 mm/s | 7–10° / 37°/s |

(The tilt after a landing includes the deliberate sideways lean of the weight shift, which the slow step holds through
the swing; the turn rate is the better measure of wobble.) The slow step lands 3–5× slower and wobbles a third as much
early in training; with more training it trades some of that softness for never falling. The quick step's landings
can't be softened with a penalty: the policy follows the reference's swing timing.

On the 80 × 45 pad the slow step doesn't work (`a3w_r80`, 30 M: 0/6 walks with six steps, 2/6 falls): it can't stand on
one foot long enough (one-foot balance on r80: 0.4 s median, §6).

GIFs (hard floor, the robot's IMU filter, seed 10000): `assets/gifs/ank3_quick_b150.gif`, `assets/gifs/ank3_slow_b150.gif`.
Foot-shaped pad vs the board (quick step, 20M): `ank3_quick_foot.gif` (hard floor, seed 10001: walks), `ank3_quick_foot_fall.gif` (seed 10000: tips forward at 0.7 s), `ank3_quick_foot_rug.gif` (medium rug, seed 10001: stands, no steps) and `ank3_quick_b150_rug.gif` (the board on the same rug and seed: 6 steps).

## 6. One-foot balance

`bal_b150` task (reference `b150A_bal1`: shift, lift the right foot, hold, lower), `scripts/balance_eval.py`:
longest time with one foot up (median over 8 walks).

| policy | model | hard | firm rug | medium rug |
|---|---|---|---|---|
| straight legs, hike lift (`a2b_r80`) | hw_ank2_r80 | 0.48 s, 1 fall | 0.22 s, 2 | 0.09 s, 3 |
| bent stance (`a3b_r80`) | hw_ank3_r80 | 0.38 s, 3 falls | 0.33 s, 0 | 0.24 s, 2 |
| the ankle study's `b150A_bal_L` | its own model (servo on the shin, b150) | 1.96 s, 2 falls | 2.13 s, 1 | — |
| the ankle study's `b150A_bal_L`, unchanged | hw_ank3_b150 (this design) | 2.25 s, 1 fall | 2.10 s, 2 | — |

The heavy foot doesn't hurt one-foot balance; the short pad does. Without ankle pitch, fore-aft balance on one foot
comes from the board's length.

## 7. Learnings

1. **Straight legs and flat feet without ankle pitch can't lift a foot.** It's a singular pose: thigh swing with a
   vertical shin doesn't change the foot's height to first order; a knee bend tilts the foot so its toe digs in (and
   pushes the robot over); a hip hike lifts it a few mm once the servos sag. Trained, the straight-leg robot never falls
   but hardly steps. Keep the knees bent at rest; a straight shin is fine if the ankle fork carries the angle.
2. **The board's length is the ankle pitch.** Every pad shorter than ~100 mm falls more, balances on one foot for
   less than 0.5 s, and learns later (its first 10 M training steps are almost all falls). Round pads fail. The
   150 × 50 board is the one that walks on every floor with no falls; the foot-shaped 96 mm pad is a distant second.
   The long boards never hit each other once the ankle keeps them flat.
3. **The reference sets the landing.** Penalizing touchdown speed (×150, ×600) didn't soften landings; slowing the
   reference's swing (1.2 s, lowering over two thirds of it, staying over the stance foot until touchdown) cut the
   touchdown speed 1.5–5× and the wobble after it by a third or more. With more training the policy traded part of that
   softness for never falling.
4. **The reward weights are fragile.** Paying a lot for lifting the foot (stride ×60, ref_swing ×15) taught policies
   to yank the foot up and fall; a progress term taught them to slide the feet; the working set is stride ×20,
   ref_swing ×6, slip −50, no progress. (I first carried the wrong set over from an unsuccessful run of the ankle study,
   which cost two rounds.)
5. **Check the open-loop reference first** (`scripts/ref_play.py`): it found the toe-dig, the sideways sliding of a
   big hike and the backward falls of a flamingo lift in minutes, before any training.

## 8. Tools

| | |
|---|---|
| `scripts/make_ankle2_model.py [--bent] [pads]` | `hw_ank2_*` (straight) / `hw_ank3_*` (bent stance) models of the designed leg, per pad |
| `slender_biped/ankle_gait.py` | straight-leg references (flat feet, hip hike, flamingo lift) |
| `slender_biped/shift_gait.py --hold-shift --rise-frac` | slow step: shift held through the swing, slow lowering |
| `scripts/ref_play.py <table> <model>` | play a reference open loop; `--gif` |
| `scripts/step_quality.py <run> [--model] [--zero]` | steps, falls, touchdown speed, tilt and turn rate after landings, slide, feet touching |
| `scripts/policy_trace.py <run>` | one walk, printed every `--every` s |
| `scripts/balance_eval.py <run> [--model]` | one-foot balance |
| `scripts/ank2_queue*.sh`, `scripts/ank3_queue*.sh` | the training queues of this round |

Runs kept (params in git): `a3s_b150` 20 M (quick step), `a3w_b150` 10 M / 20 M (slow step), `a3s_foot`, `a3s_r80`,
`a2f_b150` (straight legs). On the robot: `policy_stand.py` doesn't drive the ankle servos yet; the parts aren't
printed.
