# Round 13: fall recovery (2026-11-15 overnight)

Goal (operator, 2026-11-15 night):
- **Standing:** push resistance. When the robot is shoved hard enough that it needs a step, it should step, stabilize
  and go back to its standing pose.
- **Walking:** with an aggressive walker like k5, when the IMU sees a tip starting, it should stop walking, switch to
  recovery, stabilize, return to the standing pose and resume walking.
- **Test on the robot:** shove it while it stands, and shove it while it walks.

All runs start from `k5_high2` on its own recipe ([robot13_common.sh](../../scripts/robot13_common.sh)).

## New tools

| tool | what |
|---|---|
| `push_dv`, `push_dur`, `push_lat`, `push_walk` (mjx_env) | hand-like shoves: a horizontal force on the pelvis for 0.08-0.2 s, sized so the robot gains dv m/s. The old `push_vel` kicked the whole robot's speed, the planted feet included. |
| `sup*` (mjx_env) / [recovery.py](../../slender_biped/recovery.py) / `policy_stand --recover` | the walk -> stand -> resume supervisor, the same rule in training and on the robot |
| `sup_walker` (mjx_env) | the frozen k5 walks (filtered 0.7 as on the robot); the trained policy only acts while the supervisor has stopped the walk: a recovery specialist trained on exactly the states k5 hands over |
| [push_eval.py](../../scripts/push_eval.py) | MJX push tests: stand (one shove fwd / back / left / right at 0.1-0.4 m/s) and walk (one shove at 4-6 s while walking, with or without the supervisor) |
| [detector.py](../../scripts/robot13_analysis/detector.py) | from k5's robot logs, how early the IMU can tell a fall from a normal step |

## The detector: k5's normal steps are already near a fall

From 69 k5 robot logs (39 falls, 9.2 min of walking), predicted tilt = angle relative to the clock start + 0.1 s x the
3 Hz low-passed rate:

| threshold (roll / pitch) | falls caught | lead before 18 deg (median, p10) | false stops per minute of walking |
|---|---|---|---|
| 10 / 10 deg | 39/39 | 0.42 s (0.20) | 13.7 |
| **12 / 12 deg (default)** | 39/39 | 0.20 s (0.14) | 2.4 |
| 14 / 10 deg | 39/39 | 0.14 s (0.12) | 3.2 |

Normal k5 walking reaches 10.3 deg predicted roll at p99 and 18 deg at most. Anything earlier than ~0.2 s costs false
stops. A false stop is cheap if the recovery is safe: stand, settle, resume. But at 14 a minute it would stop on
most steps.

## Experiments

| run | idea | robot test |
|---|---|---|
| `r13a_stand` | stand only, shoves 0.05-0.3 m/s, k5's rewards (at zero command they already pay the stance pose, both pads down, standing still) | `--vx 0`: shove it |
| `r13a2_stand_step` | stand only, bigger shoves 0.1-0.4, half pure sideways, steps cheaper (`ref_contact` 3 -> 1, `scuff` -100 -> -20) | `--vx 0`: shove it |
| `r13b_walkpush` | walk + stand with shoves 0.03-0.2, no supervisor: a walker that absorbs shoves itself | `--vx 0.015`, shove; also standing |
| `r13c_sup` | walk + stand with shoves 0.03-0.25, trained with the supervisor (12 / 12 deg) | `--vx 0.015 --recover`, shove |
| `r13c2_sup_early` | as c, supervisor 10 / 10 deg, tau 0.15 | `--vx 0.015 --recover --sup-roll 10 --sup-pitch 10 --sup-tau 0.15` |
| `r13e_spec` | recovery specialist: frozen k5 walks, this policy recovers | `--run k5_high2 --vx 0.015 --recover r13e_spec` |
| baselines, no training | k5 + `--recover self` (k5 at command 0), k5 + `--recover stance` (stance pose, no policy) | |

Results below as they come in.

## Baseline: k5 as it is (push_eval, sim robot, 16 per cell; `eval_k5_baseline.txt`)

Standing (`--vx 0`), share up after one shove / share back home:

| dv m/s | fwd | back | left | right | all |
|---|---|---|---|---|---|
| 0.15 | 100 / 94 | 100 / 81 | 100 / 100 | 100 / 69 | 100 / 86 |
| 0.20 | 81 / 69 | 81 / 69 | 94 / 88 | 100 / 88 | 89 / 78 |
| 0.25 | 44 / 38 | 31 / 25 | 100 / 94 | 75 / 69 | 62 / 56 |
| 0.30 | 31 / 19 | 0 / 0 | 31 / 31 | 12 / 12 | 19 / 16 |
| 0.40 | 0 | 0 | 0 | 0 | 0 |

k5 standing absorbs up to ~0.2 m/s with its feet planted and **never steps** (0-0.9 touchdowns after the shove,
mostly a foot rocking). Fore-aft is its weak direction. 0.25-0.4 m/s is where a step is needed: the target.

Walking at 0.015 m/s, one shove at 4-6 s, share up at 14 s (all directions):

| dv m/s | no supervisor | supervisor, k5 at command 0 (`--recover`) | supervisor, stance pose (`--recover stance`) |
|---|---|---|---|
| 0 | 88 % | 91 % | 64 % |
| 0.10 | 78 % | 83 % | 47 % |
| 0.15 | 66 % | 70 % | 34 % |
| 0.20 | 48 % | 50 % | 20 % |
| 0.25 | 31 % | 33 % | 11 % |

- **Stopping and letting k5 stand helps a little** (+3 points). It stops ~0.8 times per run even unshoved, and it
  resumes walking after nearly every stop.
- **Dropping to the stance pose is much worse than doing nothing.** Putting the swing foot down and freezing the
  joints mid-step throws away the balance k5 is doing. Don't use `--recover stance` on the robot except as a
  comparison.

## r13a_stand (24M; `eval_r13a_stand.txt`)

Standing, share up / home (k5 in brackets):

| dv m/s | fwd | back | left | right | all |
|---|---|---|---|---|---|
| 0.20 | 94 / 81 | 94 / 94 | 100 / 100 | 100 / 100 | **97 / 94** (89 / 78) |
| 0.25 | 69 / 62 | 69 / 69 | 94 / 94 | 100 / 100 | **83 / 81** (62 / 56) |
| 0.30 | 12 / 0 | 0 / 0 | 62 / 56 | 56 / 44 | **33 / 25** (19 / 16) |
| 0.40 | 0 | 0 | 0 | 0 | 0 (0) |

- **Firmer, but still no steps.** It holds harder with its feet planted, and nearly every survivor is back home
  (joints within 2 deg of where they were, level, still). Touchdowns after the shove: 0-0.9, the same as k5.
- **Fore-aft is the limit** (0-12 % at 0.3 m/s): without ankle pitch, the boards give the hips little fore-aft
  authority.
- **As the recovery controller under k5** (`--recover r13a_stand`) it is no better than k5 itself at command 0 (63 %
  up against 65 %).
- **Pushes alone don't teach a step from a planted policy.** Queue 3 (alongside, from 03:33) adds stepping shaping:
  `capture_v` turns the stand terms off while the pelvis moves faster than 0.12 m/s, and `capture` pays each
  touchdown for the foot's travel along the motion. Runs: `r13f_capture` (from k5) and `r13f2_capture_a` (from
  r13a), shoves 0.15-0.45 m/s, entropy 0.01.

## All runs finished (03:00-~07:00); push tests run 2026-11-15 10:45 after the session crashed

(`eval_stand_all.txt`, `eval_walk_all.txt`, `eval_walk_k5rec.txt`; sim robot, 16 per cell)

**Standing, share up after one shove (all directions):**

| policy | 0.20 | 0.25 | 0.30 | 0.40 | touchdowns after the shove |
|---|---|---|---|---|---|
| k5 | 89 % | 62 % | 19 % | 0 | ~0 |
| r13a_stand | 97 | 83 | 33 | 0 | ~0 |
| r13a2_stand_step | 92 | 75 | **42** | 0 | 0-1 (sideways only) |
| r13f_capture | 98 | 73 | 34 | 0 | ~0 |
| r13f2_capture_a | 91 | 72 | 34 | 0 | ~0 |
| r13c_sup | **100** | 80 | 36 | 0 | ~0 |
| r13c2_sup_early | **100** | **83** | 36 | 0 | ~0 |
| r13b_walkpush | 78 | 56 | 17 | 0 | ~0 |

- **No policy learned to step.** The `capture` reward was never paid in training (0 in every eval, 30M steps): the
  policies never lifted a foot after a shove, so the shaping had nothing to reward. Every gain is planted
  resistance, about +20 points at 0.25 m/s. Back shoves stay the weak direction (0 % at 0.3 for all).
- Starting from a policy that never steps, PPO's exploration doesn't find a step on this body: a lifted foot under
  a shove falls before it can pay.

**Walking at 0.015 m/s, one shove, share up at 14 s / resumed walking:**

| setup | no shove | 0.15 | 0.25 | all | stops per run |
|---|---|---|---|---|---|
| k5, no supervisor | 88 | 66 | 31 | 62 / 62 | 0 |
| k5 `--recover` (self) | 91 | 70 | 33 | 65 / 64 | 0.8 |
| k5 `--recover r13e_spec` | 88 | 70 | 27 | 64 / 64 | 0.8 |
| k5 `--recover r13a2_stand_step` | 92 | 70 | 27 | 66 / 65 | 0.8 |
| k5 `--recover r13f2_capture_a` | 94 | 69 | 34 | 67 / 65 | 0.8 |
| k5 `--recover stance` | 64 | 34 | 11 | 35 / 35 | 0.7 |
| r13b_walkpush, no supervisor | 42 | 36 | 6 | 27 / 27 | 0 |
| r13c_sup `--recover` | 98 | 97 | 77 | 93 / **6** | 3.9 |
| r13c2_sup_early `--recover` | 100 | 98 | 75 | 94 / **2** | 4.8 |

- **The supervisor-trained walkers learned to stop walking.** r13c / r13c2 trip the supervisor ~4-5 times per run
  and walk ~1 cm: standing pays nearly as much as walking and never falls, so tipping into the stand mode became
  the policy. A reward hack. The supervisor in training needs a cost per stop, or progress pay that standing
  can't match.
- **Recovery controllers under k5 add 1-5 points**, within the noise of 16 per cell. In the sim, the stop rule
  fires ~0.2 s before the fall, too late for any standing controller to change much.
- **Training the walker with shoves (r13b) broke its walking** (42 % up unshoved against k5's 88 %).

## Robot, 2026-11-15 11:19-11:26: standing shoves (`scripts/robot13_analysis/stand_push.py`)

Standing at `--vx 0`, shoved by hand, no video. k5 ×3, r13a_stand ×4, r13c2_sup_early ×4 (plus 2 k5 walking runs
of 25 s, stopped by the operator). **All 11 standing trials ended in a fall**: 7 to the left, 4 forward (corrected 2026-11-15 13:10: the first version of stand_push.py had the roll sign backwards).

| | |
|---|---|
| time from the first visible tilt (2.5 deg) to 18 deg | 0.34-2.56 s, median ~0.55 s; slow topples 1.7-2.6 s (r13a, r13c2) |
| from 6 deg to 18 deg | median ~0.4 s |
| peak tilt rate in the falls | 68-115 deg/s |
| commanded joint motion over that window | **2-10 deg** at most, any joint; mostly hip yaw (8-12) and knee (4-10) |
| knee difference (a foot lifting) | never above 8 deg: no foot came up |

- **The policies do nearly nothing while they fall.** The simulator agrees: after a shove, the r13a survivors move
  their commands 2-5 deg, and those that fall 7-16 deg. They learned small planted corrections and nothing else. On
  the robot the window to act is real (~0.5 s, up to 2.5 s in a slow topple), and no step comes.
- The sim's planted gains (+20 points at 0.25 m/s) didn't show on the robot at the shoves given; the shoves'
  sizes are unknown (no video, no force reading).

## Research rundown (2026-11-15)

- **Reward shaping alone** (Yang, Komura, Li 2018, arXiv 1809.02074, Valkyrie model; Ferigo et al. 2021, arXiv
  2104.14534, iCub):
  - Ankle, hip and stepping strategies emerged, in simulation only.
  - Ferigo: 20M steps (~7 days of robot time), rewards on the CoM speed toward the support hull, posture, CoP and
    contacts. The steps degraded to small hops (stiff joint controllers, contact modelling). Training pushes 200 N
    for 0.2 s every ~5 s.
- **Atalante exoskeleton** (Duburcq et al. 2022, arXiv 2203.01148): standing push recovery with reactive steps on
  hardware.
  - About 190 N·s pushes survived.
  - Rewards: foot placement at the capture point, ZMP in the support polygon, balanced contact forces.
  - Training pushes: 800 N for 400 ms every 3 ± 2 s. Ramping the force up from 0 made training converge early on a
    poor policy.
  - Strict termination conditions mattered most.
  - Smoothness was imposed on the policy's mean output, not in the reward.
  - Start states came from a planner's reference motions. Tracking a reference alone was not effective (their
    references were standing and walking, not recovery steps).
- **Capture Steps** (Missura, Bennewitz, Behnke 2020, arXiv 2011.02793; small humanoids, model-based): an inverted
  pendulum fitted to the CoM picks the next footstep's place and timing. Among the strongest push recovery of any
  humanoid at the time.
- **Hybrids:**
  - Pankert et al. 2018 (KIT): learned stepping motions, with RL mapping the push to step parameters. The closest to
    our plan.
  - Egle et al. 2024 (arXiv 2411.01000): RL on top of a model-based step generator. Crossover steps matter for big
    sideways pushes.
- **Disney BD-X** (Grandia et al. 2025, arXiv 2501.05204): a separate standing policy imitating authored motion,
  trained with disturbances.
- **Open Duck** (`Open_Duck_Playground`):
  - Shoves are a velocity kick on the base, 0.1-1.0 m/s, every 5-10 s, on by default.
  - The walking env imitates a polynomial reference indexed by command and phase, with imitation off at zero command.
  - The standing env has no reference and penalizes leg motion.
  - No push-specific reward and no recovery stepping.

The pattern: real recovery steps on hardware come with a model-based step generator or a reference. End-to-end RL
without one steps only in simulation.

## Next (2026-11-15, agreed with the operator): standing push resistance first, with references

See [walking_clean.md §32](../../docs/walking_clean.md): a recovery-step reference library indexed by the step target
with its own clock, the capture point choosing the step, k5's command and clock inputs reused, large imitation rewards.
Stage A: step on command without shoves (robot check: a key triggers a step). Stage B: shoves at full size.
