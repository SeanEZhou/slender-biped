# Clean, steady steps on the ankle-roll legs (2026-11-10)

The ankle-roll legs (hardware/ankle_p1, sim model `hw_ank3_b150_legs_10dof`: 10 joints, the 150 × 50 mm board) walked
in simulation without falling ([walking_ankle.md](walking_ankle.md)), but not cleanly: the body wobbled after every
landing, the feet scuffed the floor and the planted boards slid. On the real robot a big wobble that the simulation
recovers from tends to be a fall, and a toe that catches the floor trips the robot forward. This round measures those
things and trains against them, across a range from slow and safe to bigger, more natural steps.

Everything in §1–7 is simulation (legs and pelvis, the robot's IMU filter in the loop, servo / mass / friction
randomization per walk, floors from hard to a soft rug). **On the robot (§11): `c3m_T12` is the first policy to walk,
10+ steps in most trials; the full-body `c3f_T12` balances and marches, mostly in place.**

## 1. What the open-source biped trainers do

Read for this round: [Open Duck Mini](https://github.com/apirrone/Open_Duck_Playground) (Feetech STS3215-class
servos, ~42 cm, the closest to this robot), [humanoid-gym](https://github.com/roboterax/humanoid-gym) and
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (G1).

* **A periodic gait clock and a contact schedule** in all three; the reward pays for each foot being down or up when
  the schedule says (G1 `contact`, humanoid-gym `feet_contact_number`, Open Duck the reference's contacts).
* **Imitation dominates in Open Duck**: joint tracking of a generated reference motion has by far the largest weight;
  the rest is velocity tracking and a heavy action-rate penalty.
* **Swing clearance**: G1 penalizes each swing foot's distance from 8 cm (weight -20, its largest term); humanoid-gym
  pays for reaching 6 cm.
* **Smoothness**: humanoid-gym penalizes the second difference of the actions and pays for low base acceleration.
* **Hobby-servo robustness** (Open Duck): a joint-speed limit, 0-2 steps of action and IMU delay, the last 3 actions
  in the observation.
* **Continuous gaits**: 0.6-0.8 s per cycle in all three, against 5-8 s for this robot's step-settle-step references.
* No stumble term is active in any of them.

This repo already had the clock, servo model, delays and randomization. New here: the penalties below, the contact
schedule and stronger imitation, and faster continuous references.

## 2. Measuring steadiness (scripts/step_quality.py)

Per floor, 6-8 randomized walks of 30 s, besides steps and falls:

* **wobble**: RMS body roll and pitch rate over the walk (deg/s)
* **after touchdown**: peak body tilt and roll/pitch rate in the 0.6 s after each landing
* **scuffs**: a foot moving faster than 3 cm/s within 2 mm of the floor while not planted (scraping, a catching toe),
  per step
* **touchdown tilt**: the board's pitch and roll at each landing; **toe-down** = share of landings toe first
* **slide**: travel of planted boards per step; **lift**: peak lift per step; **forward**: distance in 30 s

The best policy before this round (`a3s_b150`, hard floor): no falls, but 12 steps in 30 s, ~2 scuffs per step, the
planted boards sliding 36 mm per step, wobble ~10-12 deg/s, 6-7° of tilt after each landing, 16 cm in 30 s.

## 3. New training terms (slender_biped/mjx_env.py)

| term | what | weight used |
|---|---|---|
| `scuff` | horizontal foot speed² within 4 mm of the floor while not planted | -100 |
| `foot_tilt` | board tilt² within 2 cm of the floor: pitch (toe-down ×2) and roll while lifting / landing; roll only while planted (a staggered stance without ankle pitch tilts the boards fore-aft by design) | -10 (-30 collapsed a run) |
| `ang_acc` | body roll/pitch angular acceleration² | -0.003 … -0.01 |
| `action_acc` | second difference of the actions² | -0.1 |
| `clear` | a scheduled swing foot's shortfall below the reference lift (cm²) | -2 |
| `ang_vel_xy` | body roll/pitch rate² (existed at -0.05) | -0.5 … -2 |
| `narrow` | lateral board-centre gap below 65 mm (cm²) | -1 (did not stop touching) |
| `board_yaw` | each board's yaw relative to the pelvis heading, sin² | -100 |
| `ref_contact` | each foot up / down when the reference says | 3 |
| `ref_q` | joint tracking of the reference (existed at 1) | 3 |

**Penalties alone made it worse** (`c1_b150`, the best policy fine-tuned with them): stepping itself got expensive, so
it stepped half as often (6 steps in 30 s) and slid its boards instead (73 mm per step). The penalties only work with
the stepping anchored by imitation (`ref_contact` 3, `ref_q` 3), as in Open Duck.

## 4. References

`slender_biped/shift_gait.py --ankle`: hip-roll weight shift with the ankle rolls keeping the boards flat, lift by
equal hip-pitch and knee bends (the shin keeps its angle, so the board lifts flat, not toe-down), a settle gate that
holds the clock at each lift-off while the body still turns faster than 0.3 rad/s (also on the robot).

| table | step | weight shift | swing | lift (knee = hip) | max step | open loop (no policy) |
|---|---|---|---|---|---|---|
| `b150A_shift` (before) | 2.5 s | 1.2 s | 0.35 s | 0.3 rad | 4 cm | falls |
| `b150A_slow1` (before) | 4 s | 1.2 s held | 1.2 s | 0.3 rad | 4.8 cm | falls |
| `c_T12` continuous | 1.2 s | 0.45 s | 0.45 s | 0.45 rad (~3 cm) | 4.8 cm | walks 2 cm/s in 2/3 of randomized walks |
| `c_T08` brisk | 0.8 s | 0.3 s | 0.35 s | 0.45 rad | 4 cm | falls at 4 cm/s |
| `c_big` big step | 2 s | 0.8 s held | 0.7 s | 0.6 rad (~5 cm) | 7 cm | falls |
| `c_T15` calmer | 1.5 s | 0.55 s | 0.55 s | 0.45 rad | 5.3 cm | 1/3 walks |
| `c_T12h` high | 1.2 s | 0.45 s | 0.45 s | 0.6 rad | 4.8 cm | 2/3 walks |

## 5. Results

All at 20-25 M training steps (fine-tunes: + 20 M), 8 randomized 30 s walks per floor (some: 6), the robot's IMU
filter, commanded 2 cm/s (quick-step policies 0.8 cm/s). Hard floor / medium rug (floor tc 0.004 / 0.3). Falls are
per floor; "after landing" is the peak body tilt and roll/pitch rate in the 0.6 s after each touchdown (median).

| policy | reference | falls hard / rug | forward 30 s | touchdown mm/s | after landing | wobble roll / pitch deg/s | scuffs / step | toe-down | lift mm | boards touch |
|---|---|---|---|---|---|---|---|---|---|---|
| `a3s_b150` (before) | quick step | 0/6, 0/6 | 16 cm | 135-148 | 6-7° 45-50°/s | 10 / 12 | 1.6-1.9 | 10-17 % | 10-12 | 0 |
| `c1_b150` | quick step, penalties only | 0/6, 0/6 | - | 138-199 | 7° 55°/s | 9 / 15 | 2.1 | 12-29 % | 7-10 | 0 (slides 73-155 mm/step) |
| `c1i_b150` 10 M | quick step | 0/6, 0/6 | 23-24 cm | 115-150 | 8° 52°/s | 11 / 12 | 2.0 | 15-18 % | 6 | 0 |
| **`c3i_T12`** | continuous 1.2 s | **0/8, 0/8** | 47-56 cm | 146-223 | 7.2-7.6° 50-52°/s | 15-17 / 16-19 | 1.4-1.5 | 16-17 % | 19-20 | 2/16 walks |
| `c3s_T12` | 1.2 s, steadiness ×4 | 1/8, 2/8 | 53 cm | 117-161 | 5.5-6.1° 32-36°/s | 12-14 / 14-15 | 1.3-1.4 | 2-4 % | 22 | 1/16 |
| **`c3m_T12`** | 1.2 s, steadiness ×2 | **0/8, 0/8** | 46-55 cm | 128-199 | 5.8-6.6° 41-42°/s | 15 / 13-16 | 1.4-1.5 | 21-23 % | 19-22 | 0 |
| `c3r_T12` | 1.2 s, soft-rug training | 0/8, 1/8 | 29-31 cm | 144-183 | 7.5° 40°/s | 13-15 / 17 | 1.5-1.6 | 44 % | 17-18 | 1/16 |
| `c3p_T12` | 1.2 s, pushes in training | 3/8, 4/8 | 26-43 cm | 137-193 | 5.4-5.7° 40-43°/s | 17-29 / 16-17 | 1.3-1.6 | 6-8 % | 22 | 10/16 |
| `c7_T12h` | 1.2 s, 0.6 rad lift | 0/8, 1/8 | 43-54 cm | 130-190 | 4.6-5.8° 41-43°/s | 15-16 / 16-17 | 1.4 | 13-21 % | 30-33 | 1/16 |
| `c4i_T08` | brisk 0.8 s | 3/6, 2/6 | 38-66 cm | 112-137 | 9-10° 56-65°/s | 24-26 / 23-28 | 1.5-1.7 | 25-32 % | 14-16 | 2/12 |
| **`c6_T15`** | calmer 1.5 s | **0/8, 0/8** | 36-40 cm | 111-175 | 4.8-6.7° 39-45°/s | 16 / 13-14 | 1.5-1.6 | 13-16 % | 26-32 | 0 |
| **`c6m_T15`** | 1.5 s, steadiness ×2 | **0/8, 0/8** | 23-26 cm | 97-132 | 3.6-4.5° 30°/s | 12-13 / 12 | 1.5-1.6 | 2-6 % | 27-31 | 6/16 (≤0.5 s) |
| **`c5i_big`** | big step 2 s, 7 cm | **0/8, 0/8** | 29-35 cm | 90-145 | 5.1-6.2° 32-43°/s | 15 / 12 | 1.5 | 3-12 % | 52-58 | 2/16 |
| `c5s_big` | big step, steadiness ×4 | 0/8, 0/8 | 13-14 cm | 112-150 | 3.4-4.8° 31-33°/s | 11-12 / 11 | 1.7-1.8 | 24-26 % | 43-45 | 5/16 |
| `c2_slow` / `c2i_slow` | slow 4 s step | collapsed / 4 steps in 30 s, boards slide 13-16 cm | | | | | | 81-94 % | | |

| `c6c_T15` | c6 trained 20 M longer | 1/8, 3/8 | 32-39 cm | 107-135 | 3.7-4.2° 33°/s | 15 / 12-13 | 1.4-1.5 | 3-4 % | 33-36 | 1/16 |
| `c6n_T15` | c6m + `narrow` (centre gap) | 0/8, 0/8 | 23-24 cm | 95-111 | 3.2-3.5° 31-33°/s | 14 / 10-11 | 1.6 | 1-5 % | 34-38 (min 14) | 10/16 |
| `c6y_T15` | c6n + `board_yaw` | 0/8, 1/8 | 25 cm | 93-131 | 4.5° 28-30°/s | 15-17 / 11-12 | 1.5-1.6 | 1 % | 36-41 | 3/16 |
| **`c8_big2`** | big step at 1.6 s, 4 cm lift | **0/8, 0/8** | 32-38 cm | 113-188 | 4.3-6.4° 36-43°/s | 15-16 / 14-15 | 1.4-1.5 | 4-11 % | 38-45 | 7/16 |
| `c8m_big2` | c8 + steadiness ×2 + `narrow` | 0/8, 0/8 | 27-33 cm | 138-212 | 5.3-7.3° 42-51°/s | 17-18 / 15-16 | 1.5-1.6 | 12 % | 40-43 | 4/16 |

**Boards touching** ("boards touch" column): the contact is between the boards' collision proxies, but the boards
themselves overlap by 1-3 mm at those moments (measured with the board boxes): the boards turn toe-in, so their
ends meet while the centres stay ~55-59 mm apart. A centre-gap penalty (`narrow`) didn't help; a board-yaw penalty
(`board_yaw`, c6y) cut it from 10 to 3 of 16 walks. In 8 hard-floor walks, `c3m_T12` and `c6_T15` never touched.

**Soft rug** (floor tc 0.8, ~6 mm sink): every policy falls in 3-5 of 8 walks (c3i 4/6, c3r 5/8, c6 5/8, c5i 3/6);
training on softer rugs (c3r) did not help.


## 6. Deterministic walker (no learning)

`RefWalk` in slender_biped/rock_walk.py plays a reference table with the settle gate and IMU feedback: roll
(k_r × roll + k_w × roll rate) onto both hip rolls and, scaled by `ank`, both ankle rolls; pitch onto both hip
pitches. Gains tuned in the CPU simulator by scripts/refwalk_opt.py (cross-entropy search; 15 s walks on 4 randomized
robots × hard floor and rug; score = survival + forward distance - wobble).

| reference | gains (results/) | tuning walks (8 × 15 s) | unseen walks: falls hard / rug / soft rug (8 × 30 s each) |
|---|---|---|---|
| `c_T12` | `refwalk_c_T12_fwd.json` (forward-weighted) | 0 falls, 32 cm, wobble 38 deg/s | 7/8, 4/8, 5/8 |
| `c_T12` | `refwalk_c_T12.json` (= `_steady`; loaded by the robot runner) | 0 falls, 28 cm, wobble 30 deg/s | 3/8, 1/8, 4/8 |
| `c_T12` at 1.2 cm/s | `refwalk_c_T12_v012.json` | 1/8 falls | - |
| `b150A_shift`, `b150A_slow1`, `c_T08`, `c_big` | `refwalk_<table>.json` | 1-8 of 8 fall | - |

The tuning overfitted its 8 short walks: on new randomized robots and 30 s walks the best gains fall in 8 of 24.
The continuous 1.2 s table is the only one a fixed controller walks at all; the RL policies on the same table
(`c3i_T12`, `c3m_T12`) walk 16 of 16 without falling. Worth one careful try on the robot as a baseline (it shows
what the reference does without a network), not as the walker to rely on.

## 7. Learnings

1. **Anchor the stepping before penalizing it.** Penalties on wobble, scuffs and tilted feet alone make stepping
   expensive, and the policy steps less and slides its boards instead (`c1_b150`). With imitation weighted up
   (`ref_contact` 3, `ref_q` 3) the same penalties shape *how* it steps.
2. **Continuous gaits beat step-settle-step.** Cutting the step from 2.5 s (with a 0.35 s swing) to 1.2-1.5 s (a
   0.45-0.55 s swing, the shift and swing overlapping) gave 2-3× the speed, half the board slide, 2× the lift and
   fewer scuffs, with no falls. 0.8 s was too brisk (falls in 2-3 of 6 walks).
3. **Higher steps land more gently, as expected.** The big step (5 cm lift, 7 cm stride) and the 0.6 rad lift land
   at 90-130 mm/s with 4-5° of tilt after landing, against ~150 mm/s and 7° for 2 cm of lift. Toe-down landings drop
   to a few percent. Big steps are slower per second, but not less safe.
4. **Steadiness weights have a sweet spot.** ×2 the roll-rate / angular-acceleration penalties (`c3m`, `c6m`) calmed
   the landings with no falls; ×4 (`c3s`, `c5s`) calmed them further but fell now and then or barely moved forward.
   A foot-tilt penalty of -30 collapsed a run (`c2_slow`); -10 works.
5. **Slow 4 s steps are a dead end** with these legs: few steps, toe-down landings, sliding boards.
6. **Pushes and softer rugs in training didn't help** (`c3p` crossed its feet and fell more; `c3r` no better on the
   soft rug). Every policy still falls in 3-5 of 8 soft-rug walks.
7. **A deterministic walker with IMU feedback** walks the 1.2 s table but overfits its tuning (8 of 24 unseen walks
   fall), far behind the RL residual on the same table.
8. **Scuffs per step stayed ~1.4-1.6** for every continuous policy (down from ~2). Probably most are the lift-off
   and touchdown themselves (the board within 2 mm of the floor while already moving forward), since the median lift
   is 20-55 mm; not verified frame by frame. The lowest lift in a walk's worst step is still 3-5 mm, so a near-scrape
   happens now and then. The next thing to measure: where in the swing the scuffs happen.

## 8. Running these on the robot

The robot runner (hardware/leg_p0/software/policy_stand.py) now takes 10-joint policies (41 inputs: the 35 of the
8-joint legs + 2 ankles × 3) and the deterministic `refwalk`. The ankle servos are in the bench configs as IDs 9 (left)
and 10 (right).

**Bring-up of the ankle servos (once):**

1. Set the IDs with ONE servo on the bus (a new STS3215 answers at 1):
   `python feetech_bench.py set-id --old 1 --new 9 --i-have-only-one-servo-connected` (left ankle; 10 for the right).
2. Zero each ankle with its foot held flat (nothing moves): `python feetech_bench.py zero --id 9` and `--id 10`; or
   nudge it flat from the keyboard first: `python feetech_bench.py jog ankle_roll --enable` (right:
   `python feetech_bench.py --config bench_config_right.yaml jog ankle_roll --enable`), `z` sets the centre.
3. Robot held in the air: `python biped_bench.py play check_ankle --enable`. Each ankle rolls +10° alone: the foot's
   **outer** edge must go **up** (the simulation's sign, the same sense as hip roll + = abduction). If one goes down,
   flip that ankle's `direction` in bench_config.yaml / bench_config_right.yaml. (On the robot, 2026-11-10: the first
   guess, copied from the hip rolls, raised the inner edge on both legs; flipped to left +1, right -1.) A servo reused
   from another joint may carry old angle limits: `python feetech_bench.py info` shows them, `clear-limits --id N`
   resets them.
4. `python policy_stand.py run --run c6m_T15 --dry-run`: servos limp, tilt the robot by hand and watch the ankle
   targets move (the last number of each leg's group).

**Policies, safest first** (`python policy_stand.py run --run <run> --enable --vx <v>`; a run's newest checkpoint is
used unless `--params` says otherwise; `--vx` is required, without it they stand):

| run | gait | --vx | notes |
|---|---|---|---|
| `c3m_T12` | 1.2 s steps, 2 cm lift | 0.02 | **walks on the robot** (10+ steps in most trials); sim: 0/16 falls, boards never touched |
| `c3f_T12` | c3m fine-tuned with torso and head | 0.015 | **full body on the robot**: balances, marches mostly in place, servo buzz (§10–11) |
| `c6_T15` | 1.5 s steps, 3 cm lift | 0.02 | sim: 0/16 falls, boards never touched. **Robot: steps on its own foot right away** |
| `c6m_T15` | 1.5 s, steadier | 0.02 | calmer landings, 0/16 falls; boards overlap ~2 mm now and then (toe-in) |
| `c6n_T15` | 1.5 s, steadier still | 0.02 | calmest landings, 14 mm minimum lift; boards overlap often (10/16) |
| `c6y_T15` | c6n + boards kept straight | 0.02 | calm landings, boards touch rarely (3/16); 1 rug fall in 8 |
| `c3i_T12` | 1.2 s steps | 0.02 | 0/16 falls, more wobble than c3m |
| `c5i_big` | big step: 2 s, 5 cm lift, 7 cm stride | 0.02 | sim: 0/16 falls, gentlest touchdowns, slow. **Robot: tips about every other step** |
| `c8_big2` | 1.6 s, 4 cm lift | 0.02 | 0/16 falls; boards overlap now and then |
| `c8m_big2` | c8 + steadier, boards apart | 0.02 | worse than c8 (more wobble) |
| `c7_T12h` | 1.2 s, 3 cm lift | 0.02 | 1/16 falls |
| `a3s_b150` | quick step (before) | 0.008 | the previous best, for comparison |
| `refwalk` | deterministic, c_T12 + IMU feedback | (built in: 0.02) | `--run refwalk --ramp 0`; falls in sim 8/24 |

Start each at a lower speed (`--vx 0.012`-`0.015`, trained range 0.01-0.03) on the floor the robot stood best on.
The settle gate (the clock waits at each lift-off until the body turns slower than 0.3 rad/s) runs on the robot as in
training. Each run logs a CSV next to the runner; those logs are what to send back.

All checkpoints of this round are kept in runs/ (configs and learning curves for every run).

## 9. Session

~7.7 h of GPU (RTX 4070 SUPER): 21 training runs started, 18 run to 20-25 M steps (~21 min each), 3 stopped early (c2_slow and c2i_slow collapsed or stalled; c3_T12 restarted as c3i_T12), queued by scripts/clean_queue1-13.sh
(the queue files record every setting). Learning curves and configs for every run in runs/<name>/; the final
checkpoint of every run of this round is kept (.gitignore). New: scripts/refwalk_opt.py, the steadiness metrics in
scripts/step_quality.py, the terms in slender_biped/mjx_env.py, the references in models/ref/c_*.npz, RefWalk in
slender_biped/rock_walk.py, 10-joint support in policy_stand.py, the ankle servos in the bench configs.

## 10. Full body (torso and head on), 2026-11-10 evening

On the robot (legs only) `c3m_T12` walked with a decent success rate; `c6_T15` stepped on itself right away and
`c5i_big` tipped about every other step. So the full-body policy starts from `c3m_T12`.

Model `hw_ank3_b150_10dof` (scripts/make_ankle2_model.py --bent --full b150): the torso and head lumped into the
pelvis (0.574 kg, centre of mass 125 mm above the hips), 1.45 kg in all against 1.10 kg legs only.

| policy | falls hard / rug | forward 30 s | touchdown mm/s | after landing | wobble roll / pitch | boards touch |
|---|---|---|---|---|---|---|
| `c3m_T12` (legs only) on the full body, unchanged | 6/8, 7/8 | 3 / -13 cm | 110-124 | 6-7° 30°/s | 11-15 / 22-32 | 0 |
| `c3f_T12` 12 M | 0/8, 1/8 | 43-44 cm | 129-194 | 6-7° 28-30°/s | 11 / 9 | 1/16 |
| **`c3f_T12` 25 M** | **0/8, 0/8** | 36-38 cm | 133-198 | 6.6-7.1° 36°/s | 13 / 11 | 2/16 |

Fine-tuned from c3m_T12 with its recipe (scripts/full_queue1.sh). The extra mass high up makes the legs-only policy
pitch over; trained on it, the body wobbles less than the legs-only robot (more inertia).

On the robot: `python policy_stand.py run --run c3f_T12 --enable --vx 0.015` (newest checkpoint, 25 M; the 12 M one
with `--params params_0012M.pkl`). Result: §11.

## 11. On the robot: the first walks (2026-11-10 night)

The A01 shins and A02 feet printed, two STS3215 C001 servos on the ankles (IDs 9 left, 10 right; bring-up in §8),
5 V bench supply, IMU on the pelvis. Videos: [README §16.1](../README.md#161-on-the-robot-the-first-walks-2026-11-10).

### 11.1 Legs and pelvis

| policy | simulation (hard / rug) | robot |
|---|---|---|
| `c6_T15` | 0/16 falls, 0 board contacts | stepped on its own foot right away |
| `c5i_big` | 0/16 falls, gentlest landings | a coin flip every step whether it tips |
| **`c3m_T12`** | 0/16 falls, 0 board contacts | **10+ steps in most trials** |

This is the first walking policy of the project that works on the robot. Every earlier walker (§1 of
[walking.md](walking.md): level-sole RL walkers, step-and-settle, shift-and-step, deterministic rocks and steps) fell
on the first or second step.

### 11.2 How we got here

The path, in order, and what each step contributed:

1. **The ankle-less legs couldn't lift a foot** ([walking.md](walking.md) §1–6). On the robot ~70 % of falls came on
   the first step, toward the lifted foot. A hip-roll weight shift before the step cancelled the fall during the
   swing in 27 robot trials, but about half fell after touchdown. Analysis: with both boards down the centre of
   mass sits 15.5 mm outside either board's inner edge, a flat board fixes its shin's sideways tilt, and no pose
   within the joint limits puts the weight over one board. Every step was a fall growing by e every 0.16 s, against
   a 0.3–0.4 s swing on the real servos.
2. **Ankle roll, designed and checked in simulation first** ([walking_ankle.md](walking_ankle.md)). One STS3215 per
   leg lets the hips and ankles lean both legs together: the weight moves over one flat board. One-foot balance for
   2–3 s, and the same shift-and-step RL walker went from six steps in a quarter of walks (40 % falls) to 16 of 16
   without a fall. The design was then built (A01 shin with a 16° roll fork to keep the bent stance, A02 foot with
   the servo socket on the 150 × 50 mm board).
3. **A continuous reference** (`c_T12`, §4): 1.2 s per step, the weight shift and the swing overlapping, the ankle
   rolls keeping the boards flat, equal hip and knee bends lifting each board level. Before: a 2.5 s step-settle-step
   with a 0.35 s swing.
4. **Imitation anchoring** (`ref_contact` 3, `ref_q` 3, from Open Duck Mini, §1). The policy is paid to follow the
   reference's contact schedule and joints. Without it, the clean-step penalties made it step less and slide (§3).
5. **Clean-step penalties at the sweet spot** (§3, §7): scuffing −100, board tilt −10, roll/pitch rate −1, angular
   acceleration −0.006, action rate and second difference −0.1, swing clearance −2. `c3m` is `c3i` fine-tuned with
   these at ×2; ×4 (`c3s`) fell or stalled.
6. **Carried over from earlier rounds:** the fitted servo model with the firmware's acceleration profile
   ([servo_identification.md](servo_identification.md)), the robot's IMU filter in the loop, randomized floors
   (`dr_floor_tc` 0.02–0.8), the settle gate (the clock waits at each lift-off until the body turns slower than
   0.3 rad/s), running identically in training and on the robot.
7. **Picking the policy on the robot.** Three policies that all scored 0/16 falls in simulation behaved very
   differently on the robot, so the robot test decided. A guess at why `c3m` transferred, not proven: it spends
   the least time on one foot (0.45 s swing, against 0.55 s for `c6` and 0.7 s for `c5i`) and lifts least (2 cm,
   against 3 and 5 cm), so it relies least on balancing a single board through servo lag.

### 11.3 Full body (`c3f_T12`)

Fine-tuned from `c3m_T12` on the full-body model (§10). On the robot:

- **More stable than the legs alone.** The torso and head add inertia high up, and the body wobbles less.
- **It mostly marches in place.** In simulation it covers 36–38 cm in 30 s; on the robot little forward progress.
- **The servos buzz.** Measured from the runner's logs (two runs, 1 s windows): 30–43 % of the body's gyro signal above
  5 Hz against 7–17 % in simulation. The shake tracks the policy's step-to-step command jitter
  (correlation +0.70 / +0.77) and servo load (+0.80 / +0.70).
- **Not voltage.** The supply reads 5.2 V at rest, 5.1 V median, brief dips to 4.7 V under load. Shake correlates
  with the dips, but the range is only 3 % and the load drives both. In simulation `c3f_T12` at 5.1 V shook no more
  than at 7.4 V (gyro zigzag 4.5 against 5.5 °/s; 1/6 against 0/6 falls). 5 V does cost ~30 % of the servos'
  torque, so 7.4 V is still the plan.
- **Likely cause:** each new joint target arrives as a step every 20 ms at maximum servo acceleration; the policy's
  0.5–0.9° step-to-step jitter, absorbed by the simulator's frame and servo model, comes out as buzz on the real
  servos and compliant printed frame. Not measured directly (the logs sample at 50 Hz).
- **Next:** retrain for smoother actions (a stronger action-rate penalty, as Open Duck uses, or a low-pass action
  filter in training and on the robot), or smooth the commands on the robot side.

### 11.4 Runner changes for these runs

[`policy_stand.py`](../hardware/leg_p0/software/policy_stand.py): 10-joint policies (41 inputs); one sync read of
position, load, voltage and temperature for every servo per step, logged to the CSV, with the lowest voltage on the
status line and in the run summary; up to 3 retries per read and a ride-through of up to 5 failed reads; and on any
stop (fall, bus error, IMU silence, Ctrl-C) it **holds the stance pose with torque on** instead of going limp, so a
fault doesn't drop the robot (`--limp-on-fault` restores the old behaviour). Frequent "IMU silent" stops on this
setup came from USB/IP (WSL) stalls, not the IMU: `--imu-timeout 0.5`, or the boards on a native Linux USB port.

### 11.5 Learnings

These and every earlier round's lessons are collected in [lessons.md](lessons.md).


1. **The robot decides.** Three policies with identical simulated scores split into one walker, one that tripped
   over its own feet and one that tipped. Simulated falls are a filter, not a ranking.
2. **Margin transfers; precision doesn't.** The ankle gives the robot a real one-foot stance, the continuous gait
   keeps time on one foot short, and the transferring policy is the one asking least of single-board balance. The
   ankle-less frame needed every correction on time, and the real servos' lag made that impossible.
3. **Smoothness is a sim-to-real property.** Command jitter the simulator absorbs shows up as servo buzz on the
   robot; the action-rate penalty needs to be set for the robot, not for the simulator.
