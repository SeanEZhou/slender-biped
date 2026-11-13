# Clean, steady steps on the ankle-roll legs (2026-11-10)

The ankle-roll legs (hardware/ankle_p1, sim model `hw_ank3_b150_legs_10dof`: 10 joints, the 150 × 50 mm board) walked
in simulation without falling ([walking_ankle.md](walking_ankle.md)), but not cleanly: the body wobbled after every
landing, the feet scuffed the floor and the planted boards slid. On the real robot a big wobble that the simulation
recovers from tends to be a fall, and a toe that catches the floor trips the robot forward. This round measures those
things and trains against them, across a range from slow and safe to bigger, more natural steps.

Everything in §1–7 is simulation (legs and pelvis, the robot's IMU filter in the loop, servo / mass / friction
randomization per walk, floors from hard to a soft rug). §11 onward is the robot: what each policy did on it, and the
simulator changes that followed.

**Policies that walk on the real robot** (5 V, rug, cables from above, `--vx 0.015` unless noted; from 2026-11-12 19:26 the
cables hang slack above and slightly behind the robot instead of pulling, §18):

| policy | trained on | on the robot | where |
|---|---|---|---|
| `c3m_T12` | legs, rigid | the first walker: 10+ steps in most trials (2026-11-10); 0.015 yes, 0.02 no (11-11). 2026-11-12, 4 trials: 7-28 s, 3 / 6 / 3 / 20 steps; 2 of 4 falls sideways | §11, §14, §16 |
| `c3f_T12` | full body, rigid | balances and marches, mostly in place; 0.015 or 0.02 | §11.3, §14 |
| `s7_narrow` | pelvis + hip give k 2.4-6, stance-width penalty | 4 trials: 7-27 s (mean 18), 3-28 steps; no sideways fall | §15, §16 |
| `s8_push` | s7 + stronger forward pay | 4 trials: 8-26 s (mean 15), 4-21 steps; the longest steps (~17 mm per step, c3m ~12) | §15, §16 |
| **`s10_robust`** | s7 + give k 2-6 + IMU offsets | **4 trials: 19-28 s (mean 24), 18-24 steps: the steadiest of round 4**; one earlier walk of 93 s (95 steps) | §15, §16 |
| `b6_mid` | s10 + back lean + pull, shorter steps | 66 s, stopped; stable but mostly steps in place (11-14 mm), toes catch the rug | §17, §18 |
| **`b7_seed2`** | s10 + back lean + pull | **122 s and 99 s, 45 s**: the best leg policy so far, smooth on a flat rug; the longest steps (L 22-33, R 14-21 mm), fewest servo reversals. Toes catch on rougher rug; heels close (one fall with the heels together) | §17, §18 |
| `fb2_lean`, `fb4_soft` | full body, back lean + pull | step forward (c3f marches in place), but 5-24 s, every trial falls sideways; a 4.5-5 Hz shake on most steps | §17, §18 |

All three round-4 policies walk from the first step; their falls now come after 15-28 s and are mostly backward (§16).
Tried on the robot and failed: `c6_T15` (stepped on its own foot), `c5i_big` (tipped every other step) (§11);
`h9_w6kl_L`, `h4_w6k_L`, `h13_kl_L`, `h3_k_L` (fell sideways in the first steps, §14). Every walker before the ankle
roll fell on the first or second step ([walking.md](walking.md)). Untested on the robot: the other §12-13, §15 and §17
runs. With the cables no longer pulling (§18), all of round 5's leg policies and the s policies walk for a minute or
more; `c3m` still falls sideways within 10-20 s. The legs are now steadier than the full body, which is the gap to close
next (§18.4).

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
| `c3m_T12` | 1.2 s steps, 2 cm lift | **0.015** | **walks on the robot** at 0.015 (10+ steps, once 84 s); at 0.02 it does not (2026-11-11 evening, §14); sim: 0/16 falls, boards never touched |
| `c3f_T12` | c3m fine-tuned with torso and head | 0.015 or 0.02 | **full body on the robot**: works at either speed but steps forward little at both; balances, servo buzz (§10–11) |
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

## 12. Robot feedback round: shake, stepping in place, cables (2026-11-11)

On the robot (night of 2026-11-10): `c3f_T12` (full body) marched mostly in place, the right foot stepping less than
the left, with strong servo vibration that got worse at 7.4 V (it fell right away there; 5 V held). `c3m_T12` (legs
only) was less steady and was pulled over whenever its cables went taut. ~7 h of GPU: 19 fine-tunes of 20 M steps
(scripts/robot2_queue1-4.sh record every setting).

### 12.1 What the robot logs showed

From 24 `c3f_T12` logs ([`policy_stand.py`](../hardware/leg_p0/software/policy_stand.py) CSVs):

- **The vibration is a sideways oscillation at ~4.8 Hz.** 69-73 % of the body's roll-rate energy is in 4-6.5 Hz,
  peaking at 4.6-4.8 Hz, in nearly every run.
- **The policy drives it.** Its hip-roll and ankle-roll commands oscillate at the same frequency, in phase with the
  roll rate (coherence 0.86-0.88): a feedback loop from the gyro through the policy to the roll joints and back.
- **7.4 V makes it worse**: in the one 7.4 V run the shake (RMS second difference of the roll/pitch rate) was 17.6 °/s
  against ~12 at 5 V. Stronger, faster servos raise the loop's gain.
- **The simulator's servos filter 5 Hz too much.** Measured from command to joint position, the real joints pass
  0.25-0.41 (hip roll), 0.23-0.34 (ankle roll), 0.36-0.72 (hip pitch) of a 5 Hz command with ~30-40° less phase lag
  than the fitted model (0.17-0.28, 0.14-0.15, 0.14-0.29). In simulation the loop never oscillated.
- **The runner asks every servo for ~2230 °/s² (acceleration register 254),** while the model gives the yaw, roll and
  ankle joints the old hips' ~400 °/s² ramp. The new ankle servos honour the register, so they may be faster on the
  robot than in training.

### 12.2 What changed

| | where | what |
|---|---|---|
| `act_lpf` | mjx_env, evaluation, robot runner | action low-pass a = k·a_prev + (1-k)·a_policy; the filtered action is commanded, observed and penalized. The runner reads k from the run's config (`--act-lpf` overrides) |
| `dr_prof_acc` [0.6, 4.0] | training | servo ramps randomized up to 4× faster (was 1.5×), toward the real servos' 5 Hz response |
| `tug_force`, `tug_ramp_s` | mjx_env | a steady horizontal pull on the pelvis, random direction, on/off every 1-6 s; sudden or building over `tug_ramp_s` |
| `mirror_aug` | training (existed) | half the episodes mirrored left/right: a symmetric gait |
| `--accel-match` | robot runner | each servo's acceleration register set to the training model's ramp (yaw / roll / ankle 46 ≈ 400 °/s², hip pitch and knee 228) |
| [`scripts/robot_probe.py`](../scripts/robot_probe.py) | evaluation | per supply voltage, extra delay and pull: falls, roll energy at 4-6.5 Hz, shake, forward distance, each foot's mean step |

Evaluation: robot_probe with faster servos (`prof_acc=(25,60,60)`, the closest match to the robot's 5 Hz response),
no extra delay, 6 × 30 s at 2 cm/s, supply 0.68 (5 V) and 1.0 (7.4 V); pulls ramped over 1 s. step_quality at
2 cm/s on a hard floor and a rug.

### 12.3 Results

Shake in °/s (lower is calmer); forward in cm per 30 s; steps = mean left / right step (mm); pull = falls under a
0.5 N pull building over 1 s (5 V / 7.4 V); step test = falls hard / rug (6 walks each).

| policy | from | change | shake 5 V / 7.4 V | forward 5 V / 7.4 V | steps (5 V) | pull 0.5 N | step test |
|---|---|---|---|---|---|---|---|
| `c3f_T12` (robot's) | | | 7.7 / 10.4 | 35 / 38 | 21 / 26 | –¹ | |
| `d1_lpf_F` | c3f | filter 0.5, faster servos | 6.1 / 8.5 | 39 / 35 | 27 / 25 | | 1/6, 0/6 |
| `d2_sym_F` | c3f | mirroring, faster command | 8.2 / 10.6 | 39 / 29 | 28 / 28 | | 1/6, 0/6 |
| `d3_both_F` | c3f | d1 + d2 | 7.5 / 10.3 | 28 / 24 | 15 / 25 | | 0/6, 0/6 |
| **`d4_lpf7_F`** | c3f | d3 with filter 0.7 | **5.5 / 6.9** | 32 / 17 | 22 / 21 | 3/6 | 0/6, 0/6 |
| `d5_tug_F` | c3f | d3 + sudden pulls ≤ 1 N | 8.1 / 10.8 | 32 / 37 | 24 / 24 | 3/6 | 0/6, 0/6 |
| `e1_str_F` | d4 | stride reward ×2 | 5.1 / 6.6 | 42 / 27 | 28 / 25 | 4/6 / 3/6 | 1/6, 1/6 (boards touch) |
| `e2_ref6_F` | d4 | reference tracking ×2 | 5.5 / 7.4 | 31 / 12 | 18 / 18 | 4/6 / 5/6 | 0/6, 0/6 |
| `e3_trk_F` | d4 | speed reward ×3 | 5.3 / 7.1 | 31 / 24 | 12 / 9 (shuffles) | 2/6 / 4/6 | 1/6, 2/6 |
| `e4_tugr_F` | d4 | ramped pulls ≤ 0.6 N | 5.4 / 7.4 | 20 / 11 | 11 / 9 | 2/6 / 4/6 | 0/6, 0/6 |
| **`f1_tug15_F`** | d4 | sudden pulls ≤ 1.5 N | 5.9 / 7.7 | 29 / 20 | 16 / 17 | **1/6 / 0/6** | 0/6, 0/6 |
| **`f2_strtug_F`** | e1 | stride ×2 + pulls ≤ 1.5 N | 5.7 / 8.1 | **37 / 35** | 25 / 23 | 3/6 / 3/6 | 0/6, 0/6 |
| **`f3_stry_F` 10 M** | e1 | stride ×2 + board yaw | 5.7 / 7.9 | **62 / 60** | **44 / 39** | 3/6 / 4/6 | 1/6, 1/6 |
| `c3m_T12` (robot's) | | | 15.7 / 20.7 | 55 / 48 | 41 / 36 | 5/6 | |
| `l1_tug_L` | c3m | filter 0.5, mirroring, pulls ≤ 1 N | 15.1 / 20.9 | 35 / 29 | 25 / 18 | | 0/6, 0/6 |
| `l2_tug15_L` | c3m | filter 0.5, pulls ≤ 1.5 N | 14.5 / 20.4 | 40 / 44 | 26 / 28 | 0/6 / 0/6 | 0/6, 0/6 |
| **`e5_calm_L`** | c3m | filter 0.7, faster servos, mirroring | **8.5 / 10.4** | 32 / 23 | 23 / 21 | 2/6 / 5/6 | 0/6, 0/6 |
| **`e6_tugr_L`** | c3m | e5 + ramped pulls ≤ 0.6 N | 11.6 / 15.6 | **63 / 64** | **48 / 33** | 4/6 | 0/6, 0/6 |
| **`f4_strtug_L`** | l2 | filter 0.7, mirroring, stride ×2, pulls ≤ 1.5 N | 9.9 / 13.3 | 36 / 39 | 26 / 27 | 1/6 / 3/6 | 0/6, 0/6 |
| **`f5_tug15_L`** | l2 | filter 0.7, mirroring, pulls ≤ 1.5 N | 10.8 / 15.3 | 34 / 41 | 26 / 20 | **0/6 / 1/6** | 0/6, 0/6 |
| `g1_fartug_L` | e6 | pulls ≤ 1.5 N | 10.3 / 15.3 | 29 / 27 | 18 / 15 | **0/6 / 0/6** (0.8 N: 1/6 / 4/6) | 0/6, 0/6 |

`g1_fartug_L` is the only policy that rides out 0.8 N (on 5 V), at the cost of e6's long steps (lift 11-17 mm).

¹ not run with the ramped pull; under sudden 1 N pulls `c3f_T12` and `c3m_T12` both fall 6/6.

### 12.4 Learnings

1. **The shake is a feedback loop, and an action filter breaks it.** A low-pass of 0.7 on the actions (`d4`, `e5`)
   halves the legs-only shake and brings the full body's shake at 7.4 V below the old policy's at 5 V. 0.5 was not
   enough for the legs (`l1`). The filter is part of the policy: the runner applies it from the run's config.
2. **Asking for speed doesn't lengthen steps.** A higher command (`d2`), heavier reference tracking (`e2`, shorter
   steps) and a heavier speed reward (`e3`, shuffling) all failed. Only the per-step stride reward ×2 (`e1`, `f2`,
   `f3`) gave longer steps. On the robot, also command more: at `--vx 0.015` the reference asks for 3.6 cm steps and
   the policies take about half.
3. **Mirrored training evens the feet** (`d2`: 28 / 28 mm against 21 / 26).
4. **Cable pulls: strong sudden pulls in training work, gentle realistic ones don't.** Pulls up to 1.5 N (`l2`, `f1`,
   `f5`) raised the tolerance to a 0.5 N pull from 5/6 falls to 0-1/6; pulls up to 0.6-1 N (`d5`, `e4`) taught
   little, and gentle ramped pulls made the full body step in place. **0.8 N fells every policy but `g1_fartug_L`
   (1/6 on 5 V, 4/6 at 7.4 V)**: on one
   50 mm board the static tipping limit is about m·g·0.025 / 0.37 ≈ 1 N sideways, less while stepping. Route the
   cables from above with slack; no policy can resist a taut sideways cable.
5. **Late checkpoints can be worse** (`f3_stry_F`: 10 M walks 62 cm with 1/6 falls; 20 M falls 5/6 on the rug).

### 12.5 What to try on the robot

Run on 5 V first, `--vx 0.02`-`0.03`, cables hung from above. The filter is applied automatically.

| goal | full body | legs only |
|---|---|---|
| calmest (vibration) | `d4_lpf7_F` | `e5_calm_L` |
| all-round (calm, no step-test falls, keeps its steps at 7.4 V) | `f2_strtug_F` | `f4_strtug_L` |
| cables (0.5 N pull; g1 also 0.8 N on 5 V) | `f1_tug15_F` | `f5_tug15_L`, `g1_fartug_L` |
| longest steps (ambitious, 1/6 falls) | `f3_stry_F --params params_0010M.pkl` | `e6_tugr_L` |

`python policy_stand.py run --run f2_strtug_F --enable --vx 0.025`; add `--accel-match` to test whether the servos'
acceleration setting matters (the training model's ramps instead of 254 for all), and try 7.4 V only with the calm
ones.

## 13. Robot round 3: knee twitch, feet together, leaning over (2026-11-11 afternoon)

On the robot (2026-11-11 late morning): none of §12's policies walked more than a step. The calmest stepped on its own
other foot and tripped; the others leaned too far onto the standing leg and tipped over it. `c3m_T12` and `c3f_T12`
stayed the best, but `c3m` reached 6+ steps in only ~half of today's trials, the knees twitched constantly at 5 V,
and falls followed the boards drifting together. ~6.5 h of GPU: 18 fine-tunes of 20 M steps, one stopped at 10 M
(scripts/robot3_queue*.sh record every setting).

### 13.1 What the robot logs showed

From 47 `c3m_T12` logs (today and the night before), 34 `c3f_T12` logs and the 13 logs of §12's policies. 57 of
them were replayed through the policy offline, and the replay reproduces every logged action exactly:

- **The action filter was not involved.** It was off in every `c3m` / `c3f` run (the runner applies it only to runs
  trained with it), and today's `c3m` numbers match last night's (knee command jitter, gyro shake, foot spacing).
- **The twitch is the policy's own command jitter, not the servos.** Standing in the ready pose with the policy off,
  every joint reads perfectly still. With the policy on, every joint's target jitters 0.45-0.6° around its 100 ms
  mean; the knees and hip pitches pass about half of it to the joint (the hip rolls and ankles a fifth).
- **The jitter comes from the joint-speed inputs.** The policy's sensitivity to its inputs (`c3m`): joint angles 48,
  joint speeds 34, gravity 8 (squared, per unit). The servos report speed in 0.077 rad/s steps with 0.1-0.25 rad/s
  of noise, enough for ~2° of command jitter from the speeds alone. Training used 0.3 rad/s of speed noise;
  Open Duck uses 2.5.
- **Most falls and near-falls started at the first left step, 1.9 s after engaging.** The first right step fell
  inside the runner's 1 s blend-in (its weight shift scaled down) and landed inward (boards 81-88 mm apart instead of
  ~120), and the left foot lifted with the pelvis still 52-62 mm inside the standing board: the robot toppled toward
  the lifted foot. The others: the pelvis over or past the standing board's centre during a step, tipping over the
  support leg (as §12's policies did).
- **The real hip rolls are about 2.5× stiffer than the model's.** With one foot up, the standing hip roll sags
  1.4° under the body's weight on the robot, 3.5° in simulation (5 V); the knees match (2.7 vs 3.2°). The soft
  model hip sagged *into* the weight shift and did part of the policy's work: on the robot the pelvis lifts the other
  foot 44-59 mm inside the standing board, in simulation 21-31 mm. With the hip rolls 2.5× stiffer the simulation
  gives 37-48 mm, and `c3m`'s steps shrink to 8-12 mm at 7.4 V (marching in place) and `c3f` falls in the first
  seconds at 7.4 V, as on the robot.

### 13.2 What changed

| | where | what |
|---|---|---|
| `kp_hip_roll` | mjx_env, evaluation | hip-roll servo stiffness factor; 2.5 matches the robot (trained and evaluated with it from h3 on) |
| `noise_q`, `noise_qd` | mjx_env, evaluation | joint angle / speed noise in the observation (were fixed at 0.005 rad, 0.3 rad/s) |
| `--lcp` | scripts/train.py | Lipschitz-constrained policy ([Chen et al. 2024](https://arxiv.org/abs/2410.11825)): a gradient penalty on how fast the action changes with the sensor inputs (gravity, gyro, joint angles and speeds, last action) |
| `upright_sigma` | mjx_env | width of the upright kernel; tight = imitating the reference's level pelvis (Open Duck imitates the torso orientation) |
| `--splay` | shift_gait.py | wider stance: both hips abducted, the ankles keep the boards flat. `c_T12w6` (0.06 rad, shift 0.15): boards 138 mm apart at the ready pose (was 106); `c_T12w3` (0.03): 122 mm |
| `narrow` (existed) | training | penalty on boards closer than `narrow_min`, 10 mm under the gait's spacing |
| `--ramp` default | robot runner | 0 for ready-pose policies (they trained fully engaged from the first step); any blend now holds the gait clock until it ends |
| robot_probe | evaluation | command jitter (deg, comparable with the robot logs), board spacing, pelvis vs standing board at lift-off and its outward lean, falls in the first 5 s; `--blend` emulates the runner's blend-in |

Open Duck Mini's recipe (STS3215 servos, the closest robot to ours) next to ours: action-rate penalty -0.5 (ours
-0.1), command and IMU delay 0-3 steps (ours 0-1), joint noise 0.03-0.08 rad and speed noise 2.5 rad/s (ours
0.005, 0.3), imitation of joints, contacts *and* torso orientation and velocities. Its runtime has an optional
low-pass on the targets that training does not see.

### 13.3 Evaluation

[`scripts/robot3_eval.sh`](../scripts/robot3_eval.sh): robot_probe, 6 × 30 s at 2 cm/s, faster servos
(`prof_acc=(25,60,60)`), robot-like speed noise, **hip rolls 2.5× stiffer (the robot-like model)** at 5 V and 7.4 V,
plus 5 V with the old soft hips. Command jitter in simulation reads about twice the robot's (`c3m`: 1.0° in
simulation, 0.45-0.6° on the robot) but ranks policies the same way.

### 13.4 Results

All fine-tunes of 20 M steps. Robot-like model; jitter = command jitter (°); feet = board spacing median / minimum
(mm); lift-off = pelvis inside the standing board at lift-off (mm, the robot: 44-59); forward in cm per 30 s; steps =
mean left / right step (mm). K = robot-like hip rolls in training, N = speed noise 1.0 rad/s, LCP = Lipschitz
penalty 0.0005, smooth = Open Duck's smoothness (action rate and action jerk -0.5, command delay 0-2 steps,
`narrow`).

| policy | from | change | falls 5 V / 7.4 V | shake 5 V | jitter | feet | lift-off | forward 5 V / 7.4 V | steps 5 V | soft-hip falls |
|---|---|---|---|---|---|---|---|---|---|---|
| `c3m_T12` (robot's) | | | 0/6 / 0/6 | 21.6 | 1.03 | 125 / 82 | 37 | 57 / 28 | 27 / 26 (7.4 V: 8 / 12) | 0/6 |
| `h1_od_L` | c3m | smooth | 0/6 / 0/6 | 19.6 | 0.71 | 144 / 84 | 44 | 53 / 24 | 23 / 23 | 0/6 |
| `h2_w6_L` | c3m | smooth, wide gait, imitation ×5/3, level pelvis | 0/6 / 0/6 | 19.2 | 0.72 | 158 / 117 | 46 | 8 / 1 (in place) | 5 / 3 | 0/6 |
| `h3_k_L` 10 M | c3m | smooth + K | 0/6 / 0/6 | 20.1 | 0.82 | 129 / 87 | 28 | 55 / 48 | 38 / 34 | 0/6 |
| **`h4_w6k_L`** | h2 10 M | wide + K | 0/6 / 0/6 | 18.2 | 0.55 | 135 / 109 | 14 | 65 / 65 | 46 / 40 | 0/6 |
| `h6_nq_L` | c3m | smooth + K + N | 0/6 / 0/6 | 18.5 | 0.55 | 129 / 82 | 33 | 69 / 59 | 47 / 47 | 0/6 |
| **`h7_lcp_L` 10 M** | c3m | smooth + K + LCP | 0/6 / 0/6 | 15.8 | 0.27 | 117 / 82 | 25 | 72 / 79 | 49 / 52 | 0/6 |
| **`h9_w6kl_L` 10 M** | h4 | wide + K + N + LCP | 0/6 / 0/6 | 15.1 | 0.30 | 127 / 104 | 16 | 60 / 58 | 54 / 36 | 0/6 |
| `h12_w3kl_L` | h7 10 M | milder wide gait (w3) + K + N + LCP | 0/6 / 0/6 | 18.6 | 0.31 | 129 / 97 | 27 | 41 / 55 | 21 / 20 | 0/6 |
| **`h13_kl_L`** | h7 10 M | + N | 0/6 / 0/6 | 15.6 | 0.24 | 111 / 82 | 23 | 64 / 67 | 52 / 42 | 0/6 |
| `h14_lcp1_L` | c3m | smooth + K + N + LCP 0.001 | 1/6 / 5/6 (10 M: 6/6) | 15.8 | 0.19 | 121 / 86 | 31 | 67 / 8 | 54 / 38 | 0/6 |
| `h17_w6kl_L` | h7 10 M | wide + K + N + LCP | 0/6 / 0/6 | 18.5 | 0.26 | 136 / 114 | 25 | 61 / 53 | 26 / 33 | 0/6 |
| `c3f_T12` (robot's) | | | 2/6 / 2/6 | 11.8 | 1.02 | 141 / 81 | 48 | 41 / 33 | 26 / 24 | 1/6 |
| `h5_k_F` 10 M | c3f | smooth + K | 0/6 / 0/6 | 11.4 | 0.89 | 127 / 82 | 31 | 57 / 56 | 43 / 44 | 2/6 |
| `h8_w6k_F` | h4 | wide + K (full body) | 1/6 / 3/6 | 9.6 | 0.56 | 123 / 104 | 11 | 43 / 27 | 38 / 28 | 0/6 |
| **`h10_kl_F`** | h5 10 M | + N + LCP | 0/6 / 0/6 | 8.1 | 0.24 | 114 / 83 | 27 | 59 / 50 | 46 / 45 | 0/6 |
| **`h11_w6kl_F`** | h8 | + N + LCP | 0/6 / 0/6 | 8.5 | 0.30 | 127 / 114 | 18 | 48 / 53 | 39 / 26 | 0/6 |
| `h15_w6kl_F` 10 M | h5 10 M | wide + K + N + LCP | 0/6 / 0/6 | 9.5 | 0.31 | 144 / 114 | 41 | 54 / 57 | 41 / 25 (right foot double-steps) | 0/6 |
| `h16_kl_F` | c3f | smooth + K + N + LCP in one go | 5/6 / 5/6 (10 M: 1/6 / 3/6) | 8.4 | 0.22 | 132 / 86 | 42 | 20 / 15 | 38 / 32 | 0/6 |

Not in the table: a first try with LCP 0.002 + speed noise + stiffness randomization at once collapsed (episodes
178 steps long at 10 M; runs/zz_h2_lcp002_collapsed).

### 13.5 Learnings

1. **Read the logs before believing the story.** The filter blamed for `c3m`'s twitch was never on; replaying the
   policy against its own logs settles such questions exactly.
2. **The twitch is the policy reacting to sensor noise, and a Lipschitz penalty fixes it best.** Command jitter
   (`c3m` 1.03°): Open Duck's smoothness penalties 0.71 (`h1`); with the robot-like hips 0.80 (`h3`), plus speed
   noise 1.0 rad/s 0.55 (`h6`), plus a Lipschitz penalty 0.0005 instead **0.24-0.30** (`h7`, `h13`, `h10`) (as smooth as the filtered policies of §12, with no filter lag),
   and those policies also walked furthest. LCP is sharp-edged: 0.001 and 0.002 broke the policies.
3. **The model's hip rolls were too soft, and that hid the real weight-shift problem.** *(Half wrong, see §14: the
   servo is stiffer, but the hip as a whole gives more, invisibly to the encoders; K policies tip over on the robot.)* On the robot the pelvis
   lifts the foot 44-59 mm inside the standing board; the soft model did part of the shift for the policy (21-31 mm).
   Training with 2.5× stiffer hip rolls (K) makes the policies shift their own weight and keep their steps at 7.4 V
   (`c3m` marches in place there: 8-12 mm; K policies 20-47 mm). With K the full body takes 43-46 mm steps instead of
   `c3f`'s 24-26.
4. **A wider stance works, but not as a jump.** Switching `c3m` straight to the 0.06 rad wide gait made it march in
   place (`h2`: the swing lift collapsed at the switch); continuing from its 10 M checkpoint with K and a looser
   level-pelvis kernel walked 65 cm with the boards never closer than 104-118 mm (`h4`, `h9`), against 82 for `c3m`.
   A milder widening (0.03 rad) shortened the steps.
5. **Add one change at a time.** The same three changes reached in two stages (`c3f` → `h5` → `h10`) give the best
   full-body policy; all at once from `c3f` (`h16`) falls 5/6.
6. **Starting up matters on the robot.** Most of today's near-falls came at the very first left step, after a right
   step taken inside the runner's blend-in. The runner now engages ready-pose policies fully at once, as trained.

### 13.6 What to try on the robot

*Tried 2026-11-11 evening: all fell sideways within the first steps; see §14.* 5 V first, `--vx 0.02`, cables from above. No filter in any of these. The runner now starts ready-pose policies
without the 1 s blend (`--ramp 0` is the new default for them, also for `c3m` / `c3f`; the old behaviour was `--ramp 1`
with the clock running).

| goal | legs only | full body |
|---|---|---|
| smoothest, furthest (normal width) | `h13_kl_L`, `h7_lcp_L --params params_0010M.pkl` | `h10_kl_F` |
| wider stance + smooth | `h9_w6kl_L --params params_0010M.pkl` | `h11_w6kl_F` |
| wider stance, no Lipschitz penalty | `h4_w6k_L` | |
| robot-like hips only (control) | `h3_k_L --params params_0010M.pkl` | `h5_k_F --params params_0010M.pkl` |

`python policy_stand.py run --run h10_kl_F --enable --vx 0.02`. If the twitch is gone at 5 V, try 7.4 V with the
smooth ones: in simulation they keep their steps there.

## 14. Why §13's policies fell sideways: the legs give where the encoders can't see (2026-11-11 evening)

On the robot (19:42-19:59, 5 V, cables from above): `c3m_T12` walks at `--vx 0.015` (one walk of 84 s, one of 43 s)
but not at 0.02; `c3f_T12` works at 0.015 or 0.02 but steps forward little at both. Right after the 84 s `c3m` walk,
one trial each of `h9_w6kl_L`, `h4_w6k_L`, `h13_kl_L` and `h3_k_L` (10 M): all fell sideways over the standing leg
within the first steps (1.8-4 s). In simulation, robot-like or not, all of them walk 6/6.

Note: the first eleven `c3m` trials at 0.02 (19:45-19:49) replay exactly as `c3m`'s **10 M checkpoint**, not its
final one (a `--params params_0010M.pkl` left on the command line). The final `c3m` at 0.02 (19:50-19:51) ran 9-12 s
without falling; next to its 0.015 walks it lifted the foot with the pelvis a little closer to the standing board
(44-49 mm inside against 50-54) and rolled a little further onto it (3.3-4.4° against 3.0-3.8°): the same direction
as the h policies' falls, much smaller.

### 14.1 What the logs show

The replay reproduces every logged action exactly, so the policies did on the robot what they would do in
simulation given the same inputs. The difference is in how the body answers.

- **Every h fall is the same: the weight shift doesn't stop.** After the first right step the pelvis moves toward the
  right board at ~220 mm/s, passes its centre, the board tips onto its outer edge (+7° while the pelvis is over the
  centre, +17° at the end) and the robot rolls 21-24° over the standing leg. `c3m` makes the same shift at a similar
  speed but stops it with the pelvis 40-55 mm inside the standing board; it never balances over one board, it takes
  short steps falling gently back toward the lifted foot.
- **Same commands, different body (open loop).** The robot's logged commands played open-loop into the simulator from
  the robot's start pose: the joints follow them as on the robot (hip roll, ankle roll, hip pitch within ~1°), but the
  simulated body bounces sideways much more (roll-rate spikes 50-90°/s at each lift-off and touchdown, the robot's
  20-40°/s) and leans *inward* toward the swing foot at the first step (+5 to +9° roll) where the robot stays over its
  standing foot (-2°).
- **Same state, different outcome (branching).** The simulator started from the robot's logged state mid-fall
  (joints, joint speeds, tilt, gyro, pelvis velocity with the standing board's outer edge still) and the policy run
  from there: `h13` recovers in every seed from as late as 1.3 s (robot rolled 9°, rolling 39°/s), `h9` up to 1.8 s.
  The simulated body even accelerates *inward* from poses in which the robot kept tipping outward: with the same
  joint angles and tilt, the model's weight sits further inside the standing board than the robot's.
- **The two boards disagree.** With both feet down, the boards' tilts computed from the encoders and the IMU should
  agree; on the robot they differ by +5.2 to +5.9° (`c3m`, `h9`), in simulation by +2.3 to +3.2.

### 14.2 The mismatch: encoder-invisible sideways give at the hips

All of this fits one missing piece: under the body's weight the real legs give sideways at the hips by several
degrees, and the servo encoders don't see it (the servo holds its own output angle; what bends is beyond it, e.g. the
horns, brackets, bearings, or the hip-yaw servo carrying the leg's bending moment). It lets the real pelvis drift
further over the standing foot than the encoders say. `c3m`'s modest shift then puts the weight near the board's inner
edge, which is enough to lift a foot; the h policies' larger shift (to "over the board" by the model) carries it
past the outer edge.

Modelled in the evaluator (a patch of `CpuBiped`, not yet in the repo): the servo and the observation read q + f,
the physics uses q, with f → τ/k (10 ms lag), k in N·m/rad. 5 V, 3-6 × 15-20 s at 0.015:

| hidden give | `c3m` | `h13` | `h9` | `h4` | `h3` 10 M | `c3m` lift-off (robot -50 to -54 mm) |
|---|---|---|---|---|---|---|
| none (robot-like model of §13) | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | -35 |
| hip roll k = 3 (~5° under one-leg load) | **0/6** | **6/6** (2.2-9.5 s, all outward) | **6/6** (5 outward) | **6/6** (5 outward) | 1/6 | **-47** |
| hip roll k = 6 + ankle roll k = 6 | 0/3 | 0/3 | 0/3 | 0/3 | | -46 |
| ankle roll k = 3 only | 0/3 | 0/3 | 0/3 | 0/3 | | -39 |
| hip roll ±2.3° free play (backlash) | 0/3 | 0/3 | 0/3 | | | -40 |

With hip-roll give of k ≈ 3 the simulator reproduces the robot: `c3m` survives, every h policy falls outward over the
standing leg within seconds, and `c3m` lifts its feet with the pelvis as far inside the board as on the robot. Give at
the ankles alone, stiffer give, or free play does not.

### 14.3 What §13 got wrong

§13.1 measured the hip rolls 2.5× stiffer than the model from the encoders and trained with it (K). That is right
for the servo, but the hip as a whole is softer than either model, and the encoders can't show it. K made the
simulated body stiffer still, so the K policies learned a bigger weight shift, which is exactly what tips the robot
over. §13's evaluation never had the give in it, so it passed all of them.

All existing candidates with hip-roll give k = 3 (5 V, 6 × 20 s at 0.015; falls with direction, out = over the
standing leg; forward cm):

| policy | falls | forward | | policy | falls | forward |
|---|---|---|---|---|---|---|
| **`c3m_T12`** | **0/6** | 23 | | `h7_lcp_L` 10 M | 6/6 out | 11 |
| `c3i_T12` | 1/6 out | 20 | | `h13_kl_L` | 6/6 out | 5 |
| `e5_calm_L` | 4/6 in | 9 | | `h9_w6kl_L` | 6/6 | 1 |
| `e6_tugr_L` | 1/6 in | 36 | | `h4_w6k_L` | 6/6 | 9 |
| `l1_tug_L` | 1/6 in | 20 | | `c3f_T12` (full body) | 3/6 | 17 |
| `l2_tug15_L` | 1/6 in | 29 | | `h5_k_F` 10 M | 6/6 | 14 |
| `h1_od_L` | 1/6 in | 16 | | `h10_kl_F` | 6/6 out | 2 |
| `h6_nq_L` | 1/6 in | 32 | | `h11_w6kl_F` | 6/6 out | -7 |
| `h3_k_L` 10 M | 1/6 out | 29 | | | | |

Only `c3m` never falls, as on the robot. `e5`'s falls are inward (on the robot it stepped on its own foot). Every
policy trained with the Lipschitz penalty falls outward: smoother commands also mean slower correction. `c3f` falls
3/6 here but works on the robot, so the full body (heavier, more hip torque) probably gives less than k = 3 suggests;
the give is a calibration to measure, not yet a number.

### 14.4 Next

1. **Confirm the give on the robot** (minutes): powered in the ready pose, hold the robot by the pelvis and press one
   board sideways on the floor, or stand it on one foot by hand; watch the pelvis tilt (IMU) against the hip-roll /
   ankle-roll encoders. A few degrees of tilt with the encoders nearly still is the give; also look for where it
   bends (horn, bracket, hip-yaw servo shaft).
2. **Stiffen it mechanically if it is cheap** (bracket, horn screws, a support bearing on the hip-yaw output): less
   give helps every policy.
3. **Measure, then train with it.** [measuring.md](measuring.md) has the exercises (tip-over angles, hip give, one-foot
   stand, clearance, landing) with the simulator's prediction for each. Note: with the pelvis shell on the model
   (`hw_ank3_b150_legs_pelvis_10dof`, +110 g higher up) the give that separates `c3m` from the h policies is
   k ≈ 5 rather than 3 (`c3m` 0/4 falls at k = 6, 1/4 at 4.5; `h13` 2/4 at both): measure it. Encoder-invisible give in the training model (`mjx_env` servo: read q + f, f → τ/k, k
   randomized around the measured value at the hips and some at the ankles), K kept for the servo, and the evaluator
   with the give as the gate any policy must pass before the robot. Fine-tune from `c3m_T12`, one change at a time.

## 15. Robot round 4: the measured give in training, and outer strips (2026-11-12 night)

E1 on the robot ([measuring.md](measuring.md)): from the stance pose the robot tips sideways at 6-7 deg (the rigid
model says 14.9), its feet start slipping inward at 4-5 deg, and every sway narrows the stance (85 -> 70 -> 65 mm) and
lowers the next limit. Fore-aft is about right.

### 15.1 The give, calibrated

`scripts/tilt_test_sim.py` does E1 in simulation: a stiff "hand" (PD force at the pelvis) tilts the robot at 1 deg/s;
the tip angle is where the hand's force reverses, the slip angle where the standing board moves 2 mm. With the
encoder-invisible hip-roll give of §14 (`hidden_flex`):

| hip-roll give k (N·m/rad) | tip | slip from | gap after letting go at 6 deg |
|---|---|---|---|
| rigid | 13.8 | 4.5 | 81 -> 76 mm |
| 5 | 9.1 | 5.3 | -> 52 |
| **4** | **8.5** | **5.7** | -> 51 |
| 3 | 5.6 | 6.5 | -> 51 |
| robot | 6-7 | 4-5 | 85 -> 60-70 |

Ankle-roll give and friction change little. k = 4 is the calibration. At k = 4 the walking screen
(`scripts/robot4_eval.py`: 12 x 30 s per speed, 5 V, robot-like servos) agrees with the robot for the first time: `c3m`
and `c3f` walk, `h13` and `h9` fall 6/6. `c3m` there: 1 / 2 / 5 / 9 falls in 12 at 0.015 / 0.02 / 0.03 / 0.04 m/s,
steps ~20 mm at every speed.

### 15.2 Training with it

`mjx_env` now has the give (`hidden_flex`, `dr_flex` scales k per episode) and a `roll_lim` term. Fine-tunes on the
pelvis model, hip rolls 2.5x stiffer (as §13) plus the give:

| run | change | k=4 falls (0.015/0.02/0.03/0.04) | steps (mm) | k=3 falls |
|---|---|---|---|---|
| `c3m_T12` | baseline | 1 / 2 / 5 / 9 | 20 / 20 / 21 / 19 | 10, -, 12, - |
| `r1_flex` | give on, c3m's recipe | 0 / 0 / 1 / 11 | 19 / 20 / 22 / 4 | 1, -, 2, - |
| `r2_lim` | + roll_lim -1 past 4 deg + narrow | 0 everywhere | ~0: stopped walking | |
| `r3_long` | commands 0.02-0.04 | 0 | ~0: steps in place | |
| `s1_fwd` | r1 + progress 1, stride 40 | 0 / 0 / 0 / 2 | 22 / 21 / 23 / 26 | 1, -, 0, - |
| `s2_long` | s1 recipe, 0.02-0.04 | 0 / 0 / 0 / 0 | 28 / 26 / 24 / 25 | 4 / 1 / 0 / 3 |
| `s4_mild` | s2 + roll_lim -0.1 past 5 deg | 1 / 1 / 1 / 1 | slower | 2-3 |
| `s6_wide` | s2 + give k 2.4-6, 0.015-0.04 | 0 / 0 / 0 / 0 | 28 / 27 / 27 / 30 | 1 / 1 / 0 / 1 |
| **`s7_narrow`** | **s6 + narrow (-1 below 65 mm)** | **0 / 0 / 0 / 0** | **20 / 20 / 25 / 33** | **0 / 0 / 0 / 0** |
| `s8_push` | s7 + progress 2, stride 60 | 0 / 0 / 0 / 0 | 28 / 27 / 27 / 31 | 3 / 3 / 4 / 3 (0 on the 15 mm strip) |
| `s9_t15` | s7 on the slower c_T15 gait (1.5 s a step) | 0 / 0 / 0 / 0 (0.035) | 13 / 13 / 20 / 26 | 0 / 0 / 0 / 1; calmest (past 4 deg 6-12%) |
| `s10_robust` | s7 + give k 2-6 + IMU tilt offsets | 0 / 0 / 0 / 0 | 26 / 27 / 29 / 32 | 0 / 0 / 0 / 2 |
| `s11_pushn` | s8 + narrow x3 + give k 2-6 | 0 / 0 / 0 / 0 | 21 / 20 / 20 / 25 | 0 / 0 / 0 / 0 |

(`s3_w15`, `s5_w25`: the s2 recipe trained on the outer strips of §15.3; see there.)

Learnings:
1. **With the give, policies stop walking unless forward motion pays.** c3m's stride reward alone lets them settle on
   stepping in place (r3: as many touchdowns as c3m, speed 0), because going forward is what costs falls. A
   progress term (cycle-averaged speed / command) plus a doubled stride reward fixes it.
2. **A roll limit as a penalty hurts**, even mild (r2 stopped walking; s4 fell earlier and walked slower). The give in
   the model already makes leaning over expensive; the policy learns to stay inside it without being told.
3. **Keeping the stance wide is worth a penalty** (`narrow`): s7 keeps its gap >= 65 mm and is the first policy that
   neither falls at k = 3 nor shortens its steps.

### 15.3 Outer strips

`scripts/make_outpad_model.py` adds pads to the model. Tilt test with the give (k = 4), sideways tip: plain 8.5 deg;
a 30 mm half-disc on the outer edge 10.0 (a 45 mm one is unstable: the disc's apex is a single point, the robot
balances on it and falls fore or aft); the board moved 15 mm outward 10.3; **a full-length strip on the outer edge,
15 mm: 12.1, 25 mm: 14.5.** A straight outer edge is what sideways stability needs; a half circle gives only a point.

Walking on the 15 mm strip, untrained for it: `h13` and `h9` 6/6 -> 0/6; `c3m` 0/12 at k = 4 and 3 (10-12/12 on plain
boards at k = 3); `s7` 0/12 even at k = 2.5. Trained on the strip (`s3_w15`, `s5_w25`), policies spend the margin: they
draw the feet in and lean on the strip, and fall more or walk slower than s2 run on it. So: train on plain boards,
add the strip as a margin.

Part: `hardware/feet_p1/cad/outer_strip.py` -> `hardware/feet_p1/exports/F_outer_strip{15,25}.{stl,3mf}`. It
follows the board (120 mm flat, the 15 mm ends kicked up 25 deg), its bottom 0.3 mm above the sole (the board stands
as before, the strip touches once the board rolls outward ~1 deg; the simulator's strip is flush), and a lip over the board's top edge (7 mm past the servo
socket, 12 mm at the screws) holds it with two M2 x 6 self-tapping screws (drill the board 1.6 mm through the lip's
holes). 16 g (15 mm) / 23 g (25 mm); one part fits both feet (turned 180 deg).

### 15.4 What to try on the robot

1. `s10_robust` at `--vx 0.015`, then 0.02, 0.03 (the runner's c3m settings otherwise); `s7_narrow` as the calmer
   alternative. Compare with c3m's run: steps should be ~25-30 mm (c3m ~20), sideways sway small. At k = 2.5 (a hip
   much softer than measured) every policy here falls 3-7/12, c3m 12/12: if they fall on the robot, measure E6.
2. With the strips: `s8_push` (27-31 mm steps; 0/12 on the strip at k = 4 and 3).
3. Print two `F_outer_strip15` and repeat E1 (expect a tip near 10-12 deg instead of 6-7), then `s7` and `c3m` on them.
4. If s10 / s7 still fall sideways on plain boards, the give on the robot is softer than k = 4: E6 (`exercises.py shift`)
   measures it directly.

## 16. Round 4 on the robot: steadier sideways, now falling backward (2026-11-12 morning)

**The round in brief.** After `c3m_T12` walked (§11), about fifteen fine-tunes (§12-13: smoother actions, longer
strides, mirroring, stiffer hips, wider stances) all beat it in simulation and all fell on the robot, most within the
first steps (§14). Training didn't reliably reproduce a policy that transfers: `c3m` had been lucky, and the gap was the
simulator, not the reward. Measuring the robot (E1, [measuring.md](measuring.md)) showed why: it tips sideways at
6-7° where the rigid model says 14.9, because the hips give at larger angles where the encoders can't see it, the
lifted leg swings in and every sway narrows the stance. With that give fitted into the simulator (§15.1), the
simulator ranks the old policies as the robot does, and the policies trained on it (§15.2) all walk on the robot from
the first step and outlast `c3m` (below). The method that worked, and the plan from here: measure the robot, fix the
model until it reproduces the measurement and the robot's history, only then train. Next in line is the backward lean
(§16.1).

`s7_narrow`, `s8_push`, `s10_robust` and `c3m_T12`, four trials each at `--vx 0.015` (10:00-10:12, 5 V, rug, plain
boards), each policy's four trials started from four different orientations so the cables pull differently. Logs:
`hardware/leg_p0/software/stand_<run>_20261112_0958*`-`1012*.csv`. Every policy here walked from the first step; all
16 trials ended in a fall (the 25° tilt cut-out).

The step numbers are from the encoders and IMU: the model's forward kinematics put both boards in a level, heading
frame; a swing is a board more than 5 mm above the other, its reach the change in the boards' fore-aft offset over the
swing, halved (body advance per step). This sees neither foot slip nor the hidden give. Run through the same
estimator, simulated walks (k = 4, 6 x 30 s) read within a few mm of their true advance for `s8` / `s10` and 3-6 mm high
for `s7` / `c3m`.

| policy | trials (s) | steps | reach per step, L / R (mm) | same estimator in simulation | fell |
|---|---|---|---|---|---|
| `s7_narrow` | 20.7, 16.0, 26.9, 6.7 | 23, 16, 28, 3 | 15 / 10 | 21 / 19 | 4 backward (one back-left) |
| `s8_push` | 8.0, 11.1, 14.3, 25.6 | 4, 7, 10, 21 | 19 / 15 | 24 / 22 | 3 backward, 1 right |
| **`s10_robust`** | **18.9, 23.8, 23.6, 28.2** | **20, 18, 19, 24** | 12 / 6 | 23 / 23 | 3 backward, 1 left |
| `c3m_T12` | 7.4, 11.0, 7.9, 28.1 | 3, 6, 3, 20 | 13 / 11 | 19 / 20 | 1 backward, 2 sideways, 1 back-left |

What the logs show:

1. **The sideways falls are mostly gone.** `c3m` fell sideways (|roll| > 20° at the cut-out, little pitch) in 2 of 4
   trials, and leaned left in a third; the new policies in 2 of 12 (`s8` once right, `s10` once left), `s7` never. The
   give in training (§15) did what it was for.
2. **The new failure is backward.** 11 of 16 trials end pitched back 16-25°. It is not sudden: the pitch,
   per 2.4 s gait cycle, settles at 0 to +2° back after the first cycle (the robot stands at +3.8, E1's back
   lean), then creeps to +4 to +6
   over the last 2-3 cycles, and the fall comes during a swing. In simulation the same policies walk at 1-3° *forward*
   and hold it. So the robot walks 3-6° further back than the model, and drifts further back the longer it walks.
3. **`s10_robust` is the steadiest**: every trial 19-28 s, mean 24 (`s7` 18, `s8` 15, `c3m` 14); it also walked 93 s
   (95 steps) in an earlier trial at 09:54. `c3m` reached 20 steps once, in its fourth trial.
4. **Steps are shorter than in simulation, and lopsided.** By the encoders the robot reaches 40-75% of the simulated
   steps, and every new policy steps further with the left foot than the right (`s10` 12 / 6 mm); in simulation they
   are even. Of the four, only `s8` steps clearly longer than `c3m` on the robot (~17 against ~12 mm). `s8` also holds
   each foot up 1.0-1.2 s of its 1.2 s step (0.7-0.8 in simulation), `s7` and `s10` only 0.45-0.5 s (0.6-0.8).
5. **The orientation mattered.** For `s8`, `s10` and `c3m` the fourth orientation gave the longest walk (25.6, 28.2,
   28.1 s), for `s7` the shortest (6.7 s). Since each policy's four trials ran in order, orientation and trial order
   can't be told apart, but a backward pull growing with distance walked fits a cable from above and the pitch creep
   in 2.
6. **Not heat.** Hip-pitch and knee servos read 23-28 °C throughout.

### 16.1 Next

1. **Put the back lean in the model.** The robot stands 3.7° back with its boards flat (E1), and walks 3-6° further
   back than the simulator; there is no ankle pitch to correct it. In training: hip-pitch zero offsets that lean the
   real body back 0-6° (the policy's encoders reading the stance angles), and a small backward pull at the pelvis
   (a cable). `s10`'s IMU tilt offsets only fake the reading, not the lean.
2. **On the robot, separate the cables from the lean**: a couple of `s10` trials with the cables slack (held and
   followed by hand), and note which orientation each trial uses.
3. **The left / right asymmetry**: check the hip-pitch and knee zero offsets of the right leg (the right foot reaches
   half as far as the left in `s7` and `s10`).

## 17. Round 5: the backward lean in the model, and the full body (2026-11-12 afternoon, simulation)

No new measurements this round: the simulator was fitted to §16's 16 robot walks, then trained on. Scripts:
`scripts/gap_fit.py` (runs policies in the robot-like evaluator and reports what the robot logs show, measured the same
way: time to the fall, fall direction, walking pitch, step reach per foot from the encoders; `--logs` does the same for
robot logs), queues `scripts/robot5_queue{1..4}.sh`, notes and raw results in `results/robot5/`.

### 17.1 What the walks say about the model

- **The servos aren't it.** On the robot hip pitch and knee follow their commands within 1-2° (80-140 ms lag, knees
  sagging 1.5° under load), and the joint ranges while walking match the simulator's (hip pitch ~23°, knee ~25°).
- **The falls happen at landings.** Each swing lifts the thigh to 35-38°; at touchdown the body pitches back (up to
  -96°/s, to +8-14°) and, late in a walk, doesn't come back.
- **The robot walks leaning back.** It stands 3.7° back of the model with its boards flat (E1) and walks at +1 to +5°;
  the simulator at -1 to -2.

Three changes to the evaluator (`EvalConfig`) and the training environment (`EnvConfig`):

| change | what | effect |
|---|---|---|
| `hidden_off` | encoder zero errors the servos can't see; knee -0.045 rad stands the model 3.7° back as the robot | walking pitch +3-4° and pitch wobble (7-8° p5-95) as on the robot; alone, few falls |
| `drag_back`, `drag_back_max` | a backward pull at the pelvis growing with the distance walked (a cable from above, rug drag): N per m, capped | 2 N/m capped 0.6 N: everything falls in 10-16 s (too strong) |
| hip-roll give k 3.5 (was 4) | softer sideways give | c3m's sideways falls appear |

**Fitted simulator: hip-roll give k = 3.5, knee offset -0.045 rad, pull 0.6 N/m capped at 0.2 N.** It reproduces the
robot's ranking and its backward falls (12 x 30 s each, 0.015 m/s):

| policy | robot: mean time to fall | fitted sim: mean time (falls) | sim: fall directions |
|---|---|---|---|
| `s10_robust` | 23.6 s | 24.8 s (5/12) | back |
| `s7_narrow` | 17.6 | 21.1 (7/12) | back, one left |
| `s8_push` | 14.8 | 19.8 (8/12) | back |
| `c3m_T12` | 13.6 | 17.2 (9/12) | back 5, left 4 |

Not fitted: the robot's steps are still about half the simulated ones. The joints move as far; the swing is shorter
by the encoders (0.45-0.5 s against ~0.6), which points at the foot touching down early on the rug: less real clearance
than the model has (E4 measures it).

### 17.2 Training on it

Fine-tunes with the round-4 recipe plus, randomized per episode: knee offset U(-0.07, 0) common to both legs plus
±0.015 rad per joint, hip pitch ±0.015 per joint (the legs differ), pull 0-1 N/m capped 0.3 N. Yardstick: the fitted
sim (24 seeds), and a harder version (give k 3, knee -0.07, pull 1 N/m capped 0.3 N).

Legs (`hw_ank3_b150_legs_pelvis_10dof`):

| run | from, change | fitted sim | harder | robot-like k=4 (old screen): falls, steps |
|---|---|---|---|---|
| `s10_robust` | round 4 | 5/12, 36 cm | 11/12 | 0/12, 26-32 mm |
| **`b1_lean`** | s10 + lean + pull | **1/24, 50 cm** | 4/12 | 0/12, 18-20 mm |
| `b2_lean_s8` | s8 + lean + pull | 0/12, 67 cm | 8/12 | 1/12 (at 2 s), chatter |
| `b3_lean_long` | b1 + progress 2, stride 60 | 8/12 | 11/12 | 0/12, 29-33 mm |
| `b4_lean_soft` | b1 + softer give, stronger lean / pull | 6/12 | 12/12 | 0/12, 26-28 mm |
| `b5_seed1` | b1's recipe and start, training seed 1 | 10/24 | 12/12 | 0/12, 28-32 mm |
| **`b6_mid`** | b1 + progress 1.5, stride 50 | **0/24, 52 cm** | **2/12** | 0/12, 19-20 mm, calm (7-11%) |
| **`b7_seed2`** | b1's recipe and start, seed 2 | **1/24, 60 cm** | 6/12 | 0/12, 27-29 mm |
| `b8_mid2` | b7 + progress 1.5, stride 50 | 3/24, 67 cm | 9/12 | 0/12, 26-28 mm, calm (9-11%) |

Full body (`hw_ank3_b150_10dof`, torso and head):

| run | from, change | fitted sim | harder | robot-like k=4: falls, cm / 30 s, steps |
|---|---|---|---|---|
| `c3f_T12` | round 2 | 7/12, 22 cm | | 0/12, 44-53 cm, 14-19 mm |
| `fb1_s10` | s10's recipe on the full body | 8/12, 30 cm | | 0/12, 55-60 cm, 20 mm; sways (|roll| > 4° 27-34%) |
| **`fb2_lean`** | fb1 + lean + pull | **2/24, 43 cm** | 6/12 | 0-1/12, 36-44 cm, 15-17 mm, calm (12%) |
| `fb3_c3f` | c3f + give + lean + pull | 2/12, 38 cm | 10/12 | 0/12, 44 cm, 16 mm |
| `fb4_soft` | fb2 + softer give, stronger lean / pull | 4/24, 47 cm | 6/12 | 0/12, 58-59 cm, 20-21 mm |
| `fb5_seed1` | fb2's recipe and start, seed 1 | 12/24, 49 cm | 12/12 | 0/12, 67-79 cm, 24-27 mm |

Learnings:
1. **The old screen can't see the lean.** `b3` and `b4` look like the best walkers yet without it (0/12 falls, 26-33
   mm steps) and fail the fitted sim; `b1` and `fb2` pass it. Ranking against the fitted simulator is the point of
   fitting it.
2. **More forward pay spends the margin** the lean takes away (`b3`), and harsher training conditions don't
   automatically give more margin (`b4` came out worse than the run it started from).
3. **Policies trained with the lean lean forward to meet it**: `b1_lean` walks at -3° in the unfitted simulator, +0 to
   +1° in the fitted one (s10: +3 to +4°, falling back).
4. **The recipe doesn't reliably reproduce its best result.** Run again with another training seed, `b1_lean`'s
   recipe gave `b5_seed1` (10/24 in the fitted sim against 1/24), and `fb2_lean`'s gave `fb5_seed1` (12/24 against
   2/24); seed 2 (`b7_seed2`) passes again (1/24) and with s10-length steps. So the leg recipe lands on 2 of 3
   seeds. As with `c3m`, which policy a run ends up with is partly luck; the difference now is a simulator that tells
   the lucky ones apart before the robot does.

### 17.3 What to try on the robot

1. Legs: **`b6_mid`** at `--vx 0.015`, then 0.02 / 0.03 (`b1_lean` the alternative). Expect walks well past s10's
   24 s, the pitch staying near 0-2° instead of creeping to +5, and steps no longer than s10's. Then **`b7_seed2`**
   for longer steps (27-29 mm in the old screen, s10's length; 1/24 in the fitted sim, less margin in the harder one).
   If they fall back as s10 did, the fitted lean / pull is too weak; if they fall forward, too strong.
2. Full body: **`fb2_lean`** at 0.015 (`c3f_T12` for comparison); `fb4_soft` for longer steps.
3. Measurements, to replace guesses in the fitted simulator: one trial with the cables slack (is the pull real?); E4,
   foot clearance (the short steps); E6, the hip give directly (k 3.5 or 4?); the knee / hip-pitch zeros of each leg
   with the boards flat (the lean, and the L / R reach difference).

## 18. Round 5 on the robot: cables off the robot, a centred stance, and the full body shaking (2026-11-12 evening)

Logs `hardware/leg_p0/software/stand_<run>_20261112_1926*`-`2123*.csv`. 5 V, rug, `--vx` 0.015 or 0.02 (the logs can't
tell the two apart). Analysis scripts: the step estimator of §16 (`scripts/gap_fit.py`), plus the swing board's tilt
and the 4-8 Hz content of the gyro and commands.

### 18.1 The cables were part of the problem

Earlier in the evening the cables still hung from the side and pulled on the robot. Depending on the direction the
robot faced, they helped it or held it back, so trials of the same policy disagreed. From 19:26 (the last `b6_mid`
trial) the cables hover above the robot, slightly behind it, and stay slack. The results got much more consistent. Only
those runs are used below.

This also changes §17's model. The backward pull fitted there (0.6 N/m capped at 0.2 N, trained on 0-1 N/m) was mostly
these cables. The knee offset that stands the model 3.7° back is real: without any pull, `s10` still drifted back and
fell on its back (18.2).

### 18.2 Legs

| policy | trial | how it ended | pitch, early -> late (+ = back) | step reach L / R (mm) |
|---|---|---|---|---|
| `b6_mid` | 66 s | stopped | -6.7 -> -6.2 | 11 / 14 |
| **`b7_seed2`** | **122 s** | stopped | -4.9 -> -3.9 | 24 / 14 |
| `b7_seed2` | 99 s | fell right | -4.6 -> -2.5 | 22 / 18 |
| `b7_seed2` | 45 s | fell left | -3.9 -> -3.3 | 33 / 21 |
| `c3m_T12` | 17 s, 9 s | both fell right | -2 -> 0 | 1-13 |
| `s10_robust` | 105 s | fell back | -0.9 -> **+7.5** | 9 / 10 |
| `s8_push` | 47 s | stopped | -0.9 -> +1.5 | 22 / 20 |
| `s7_narrow` | 90 s | stopped | -1.5 -> +2.6 | 16 / 13 |

What was seen on the robot, and what the logs show:

1. **Every new policy is stable; none falls on its first steps.** With the measured hip give, the knee offset and the
   cables out of the way, policies that walk in the simulator now also walk on the robot. `c3m` is the exception: its
   sideways stability is clearly worse, and both its trials fell right within 20 s.
2. **The b policies stand centred, slightly forward.** They walk pitched 4-7° forward and hold it. They were trained to
   expect the 3.7° back lean *and* a backward pull. Without the pull they over-correct a little. The s policies drift
   back (`s10` to +7.5° before it fell), and even a slight push from the front makes them fall on their backs. The b
   stance is the one to keep.
3. **The cost of leaning forward: the toes catch the rug.** The robot has no ankle pitch, so the board's pitch is the
   body's pitch. Leaning forward tilts the swing board toe-down, and the toe lifts last and lands first. By the
   encoders and IMU (both boards' tips, rigid legs):

   | policy | swing board toe-down at peak lift | swings with the toe under 5 mm through mid-swing |
   |---|---|---|
   | `b6_mid` | +7.3° | 29% |
   | `b7_seed2` | +3.7 to +4.7° | 0-5% |
   | `c3m_T12` | +3.3° | 14% |
   | `s8_push` | +2.3° | 0% |
   | `s10_robust` | +0.9° | 6% |
   | `s7_narrow` | -2.0° | 5% |

   The real clearance is lower still: the hidden give and the rug's pile aren't in these numbers. On the robot, most
   toe catches don't trip it. Instead the step comes to nothing, a none-step, and the robot stays in place. When `b6`
   and `b7` do fall, a caught toe is usually the cause. `b6` leans furthest and mostly steps in place. `b7` leans less
   and steps harder, so it catches less. On a flat rug it walks smoothly. On a rougher rug many of its steps come to
   nothing.
4. **`b7` turns its toes out, and its heels come close.** Its hip yaws sit near +6° (s8 +2°), so its toes are
   about 20 mm further apart than its heels. Its heel gap (heel-tip centres, by the encoders) has a median of 107-111 mm
   and drops to 88-92 mm, the narrowest of all (`s7` / `s8` 106-112 at their narrowest). The encoders can't see the
   hip give, which narrows the stance by another 20-30 mm (E1), so only 10-15 mm can remain between the heel edges. On
   the robot the heels sometimes scrape. Once they came together, and the robot fell at once.
5. **`b7` and `s8` take the longest steps:** about 20-30 mm per step, against 9-16 mm for the others. `b7` still steps
   further with the left foot than with the right, as every policy did in §16; `s8` is even. On the robot `s8` is the
   best of the s policies and close to `b7`.
6. **The b policies switch their servos less.** `b7`'s knee loads reverse direction 2-3 times per second, against
   3.3-5.6 for the s policies. Its command jitter is about 20% lower (second difference 0.65-0.72° against
   0.80-0.88°). A body leaning forward likely keeps the knees loaded in one direction.

### 18.3 Full body: steps, but falls sideways and shakes

`fb2_lean` and `fb4_soft` (§17.2), in seven trials (20:03-20:04 and 21:21-21:23):

| policy | trials (s) | how they ended | pitch | step reach L / R (mm) | lift, median (mm) |
|---|---|---|---|---|---|
| `fb2_lean` | 12.3, 23.8, 5.4, 15.2 | all fell sideways (2 right, 2 left) | -2 to -5 | 7-23 / 6-17 | 19-27 |
| `fb4_soft` | 6.0, 10.5, 18.8 | all fell left | -1 to -6 | 7-16 / 11-15 | 32-45 |
| `b7_seed2` (legs, for comparison) | 122, 99, 45 | | -3 to -5 | 22-33 / 14-21 | 26 |

They step forward, which `c3f` never really did, but they are far less stable than the legs alone:

1. **Each step decides between tipping and recovering.** The full body rolls ±5-6° on every step; `b7` rolls
   ±1-3°. That is right at the sideways limit measured in E1 (5-7° before the feet slip in). Every trial ended in a
   sideways fall.
2. **A 4.5-5 Hz shake through the whole body, in roll, pitch and yaw.** The gyro's 4-8 Hz component, as RMS:

   | | robot: roll / pitch / yaw (°/s) | robot: share of the commands' power at 4-8 Hz | simulator (fitted, no pull): roll |
   |---|---|---|---|
   | `fb2_lean` | 24-31 / 20-22 / 27-31 | 8-15% | 7 |
   | `fb4_soft` | 16 / 16 / 21 | 3% | 6 |
   | `c3f_T12` (2026-11-11 logs) | 25 total, peak at 5.0-5.5 Hz | | |
   | `b7_seed2` (legs) | 6.5 / 5 / 11 | 1% | 10 |

   The legs match the simulator; the full body shakes 3-4 times more on the robot. `c3f` had the same shake (§11.3's
   "buzz", now located at 5 Hz). Within a run, a swing's shake follows the policy's 4-8 Hz command content (correlation
   0.83), with the hip yaws, knees and ankle rolls taking part. So the policy is feeding a body oscillation that the
   simulator doesn't have. The servo loop isn't the cause: it runs on time (3 ms per cycle, under 0.2% of cycles late).
3. **What makes the clean steps clean** (67 swings across the trials, the calmest quarter against the shakiest
   quarter, by 4-8 Hz gyro):

   | | 4-8 Hz gyro during the swing | already before lift-off | roll at lift-off, toward the standing foot | lift | swing time |
   |---|---|---|---|---|---|
   | calmest quarter | 18°/s | 26 | **+3.4°** | 41 mm | 0.82 s |
   | shakiest quarter | 50°/s | 49 | **-0.1°** | 21 mm | 0.47 s |

   A clean step starts with the weight already on the standing foot. A shaky one lifts a foot that is still carrying
   weight, so the lift comes out short and low, and the shake from the previous step carries on. The falls are the
   other extreme. In every trial the last swing starts with the body already rolled 5-12° over the standing foot,
   and it tips outward. The weight shift is either too small (shake) or too large (fall). Somewhere in between is the
   "one clean step in four".
4. **The simulator has no such mode.** Pushed sideways or fore-aft while holding its stance, the simulated full body
   doesn't ring at all: the motion dies within 0.5 s at 0.1-1 Hz. Softening the hip give to k = 2, adding a 40 ms
   delay or weakening the servos doesn't produce a 4.5-5 Hz shake either. The model lumps the torso and head rigidly
   into the pelvis (§10: 0.574 kg, centre of mass 125 mm above the hips). The real torso is printed in sections, with
   the head on top. Their own flex, the pelvis mount and the hip give under 1.45 kg can make a lightly damped mode near
   5 Hz. A policy trained without it pushes at the wrong phase.

### 18.4 Next: measure the full body

The goal is walking with the full body, and it is the full body that the simulator misses now. As in round 4, the
plan is to measure first, fit the model, then train:

1. **Tap test (new; the 5 Hz mode).** Under `hold`, tap the head sideways, then fore-aft, then twist the torso
   about the vertical. Do each a few times and log the gyro. Two numbers come out, the mode's frequency and how fast it
   dies out, which together give stiffness and damping. Repeat with the robot held by the torso (feet free), which
   isolates the hips and legs from the torso's own flex. Simulator: no ringing.
2. **E0 / E1 with the torso and head on.** Weight against the model's 1.45 kg; tip angles forward, back and sideways.
   The sideways limit is the margin each step has. With the centre of mass higher, it should come out under the
   legs-only 5-7°.
3. **E6 / E3 with the full body:** the hip give under the larger load (k = 3.5 was fitted on the legs alone), and how
   far the weight can shift onto one board before it tips outward. That bounds the shift that 18.3.3 shows the policy
   getting wrong.
4. **E4, foot clearance** (for the legs' toe catches).

In the model and training, once measured:
- a torso joint with the measured stiffness and damping (or the measured give at the hips);
- the pull reduced or removed;
- for the legs, starting from `b7`:
  - a swing-clearance term on the board's front tip (toe-down near the floor);
  - the stance-width penalty applied to the heel corners as well as to the board centres.
