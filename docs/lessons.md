# Lessons learned (start here)

What this project has learned, in one place, for whoever picks it up next: me later, or an agent. Each lesson is a
short statement plus a link to where it was learned, with the data. The detail docs keep the full record; this file
keeps the conclusions. **Add to it whenever a round ends.** Last updated 2026-11-11, after robot round 3
(knee twitch, feet together, weight shift).

Status in one line: the robot (10 STS3215 servos, ankle roll, 150 × 50 mm snowboard pads, IMU on the pelvis) walks
with the legs alone (`c3m_T12`, 10+ steps in most trials) and marches mostly in place with the torso and head on
(`c3f_T12`, servo buzz). See [walking_clean.md §11](walking_clean.md#11-on-the-robot-the-first-walks-2026-11-10-night).
Two retraining rounds since: §12's filtered policies and round 3's (§13) all failed on the robot, falling sideways
over the standing leg. Why ([walking_clean.md §14](walking_clean.md#14-why-13s-policies-fell-sideways-the-legs-give-where-the-encoders-cant-see-2026-11-11-evening)):
the real hips give sideways under load where the encoders can't see it; with that in the simulator only `c3m`
survives, as on the robot. `c3m` walks at `--vx 0.015`, not 0.02.
Next: measure the robot against the simulator ([measuring.md](measuring.md)) before more training.

## 1. The central lesson: design for margin, not precision

- **A frame that works in simulation only through constant, precise corrections won't transfer.** The tip foot and
  the ankle-less big pads both stood and walked in simulation. On the robot, neither could walk. Servo lag, backlash,
  the firmware's slow motion profile, IMU error and the floor all make some corrections late, and on a frame with no
  slack one late correction is a fall. ([README pivot](../README.md#where-this-project-stands-and-the-pivot-to-ankles),
  [sim_to_real.md §3](sim_to_real.md#3-what-we-learned), [walking.md §1](walking.md))
- **Give the robot a stance that holds while the controller is late or wrong.** That is what the ankle roll servo
  bought: a real one-foot stance. Of three walkers that all scored 0/16 falls in simulation, the one that worked on the
  robot spends the least time on one foot (a guess, not proven). ([walking_clean.md §11.2](walking_clean.md#112-how-we-got-here))
- **Simulated falls are a filter, not a ranking.** Policies with identical simulated scores split on the robot into a
  walker, one that stepped on its own foot, and one that tipped. Plan to try several on the robot.

## 2. What led to the first walk

> **In one line:** give the robot a stable one-foot stance (the ankle), then teach it by **imitating a continuous
> gait, joints and foot up/down timing together**, and only then add penalties to clean the steps up.

| # | change | why it mattered | evidence |
|---|---|---|---|
| 1 | **Ankle roll servo** (one STS3215 per leg) | Without it, every step was a sideways fall. With it, the weight moves over one flat board. | Sim: 6+ steps in ¼ of walks → 16/16. Robot: ankle-less legs fell on the first step. |
| 2 | **Imitate a continuous gait** (weight shift and swing overlapping, 1.2 s per step) | Less time on one foot; the policy has a smooth, complete example to follow. | 2–3× the speed, half the board slide, 2× the lift, no falls in sim. |
| 3 | **Imitate the foot up/down timing too** (`ref_contact` 3, with joint tracking `ref_q` 3) | Anchors the stepping, so the policy can't trade steps away for safety. | Without it, penalties made it step half as often and slide (`c1_b150`). |
| 4 | **Clean-step penalties, moderate strength** (wobble, scuffing, tilted landings, jerky actions) | Shape *how* it steps once the stepping is anchored. | ×2 calmed landings with no falls; ×4 fell or stalled. |
| 5 | **Choose on the robot** | Three policies with 0/16 sim falls behaved completely differently on the robot. | Only `c3m_T12` walked. |

**Key training lesson: imitation first, penalties second.** Penalties alone make stepping expensive, and the policy
learns to step less. Imitating a smooth, continuous gait (joints *and* foot contacts) keeps it stepping; the
penalties then refine the steps.

All of these follow §1: each one adds room for error rather than asking for precision. The ankle and the training
were never separated on the robot, so their relative share is inferred from simulation.

<details>
<summary>Why the ankle was needed (detail)</summary>

With both boards down, the centre of mass sits 15.5 mm outside either board's inner edge. A flat board locks its
shin's sideways tilt, so no pose within the joint limits puts the weight over one board, and every step is a sideways
fall growing by e every 0.16 s against a 0.3–0.4 s swing. A hip-roll weight shift helped only partly on the robot (27
trials: it cancelled the fall during the swing, about half still fell after touchdown). With the ankle, the hips and
ankles lean both legs together and the weight moves over one flat board.
([walking.md §1, §6, §9](walking.md), [walking_clean.md §11.2](walking_clean.md#112-how-we-got-here))
</details>

### Techniques borrowed from open-source biped trainers

From [Open Duck Mini](https://github.com/apirrone/Open_Duck_Playground) (STS3215-class servos, ~42 cm, the closest
to this robot), [humanoid-gym](https://github.com/roboterax/humanoid-gym) and
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (G1). Survey: [walking_clean.md §1](walking_clean.md#1-what-the-open-source-biped-trainers-do).

| technique | from | here | result |
|---|---|---|---|
| **Imitation as the anchor**: track a reference motion's joints | Open Duck | `ref_q` 3 (was 1) | **The key one** (row 3 above). |
| **Foot contact schedule**: each foot up / down when the gait says | all three | `ref_contact` 3 | Used in the best policies. Part of the anchor. |
| **Short continuous gait cycles** | all three (0.6–0.8 s per cycle) | 2.4 s per cycle (was 5–8 s) | Worked (row 2). Faster (1.6 s) fell in 2–3 of 6. |
| Second difference of the actions | humanoid-gym | `action_acc` −0.1 | Used; effect not tested alone. |
| Swing clearance | G1, humanoid-gym | `clear` −2 | Used; with `scuff`, scuffs ~2 → ~1.5 per step. |
| Heavy action-rate penalty | Open Duck | `action_rate` −0.1 | ~5× lighter than theirs; going heavier is the next fix for the servo buzz. |
| Hobby-servo robustness (delays, speed limit, past actions as input) | Open Duck | from earlier rounds | Carried over, not new. |

The gait reference itself is ours, procedurally generated for this robot (`slender_biped/shift_gait.py --ankle`):
hip-roll weight shift, ankle rolls keeping the boards flat, equal hip and knee bends to lift each board level.
Our own terms: `scuff`, `foot_tilt`, `board_yaw`, `narrow`, `ang_acc`, and the **settle gate** (the gait clock waits
at each lift-off until the body turns slower than 0.3 rad/s, in training and on the robot).
Definitions: [walking_clean.md §3](walking_clean.md#3-new-training-terms-slender_bipedmjx_envpy).

## 3. Training recipe: what works and what doesn't

The working recipe for the ankle legs (`scripts/full_queue1.sh` and `scripts/clean_queue*.sh` record every setting):
`progress 0, stride 20, ref_swing 6, slip −100, ref_contact 3, ref_q 3` plus `scuff −100, foot_tilt −10,
ang_vel_xy −1, ang_acc −0.006, action_rate −0.1, action_acc −0.1, clear −2`, randomized floors `dr_floor_tc
[0.02, 0.8]`, warm-started from the previous best.

- **Imitate a continuous gait, joints and foot contacts, before penalizing anything** (§2). Penalties alone make
  the policy step less and slide. ([walking_clean.md §7](walking_clean.md#7-learnings))
- **Penalty weights have a sweet spot.** Steadiness ×2 calmed landings with no falls; ×4 fell or stalled. A board-tilt
  penalty of −30 collapsed a run; −10 works. ([walking_clean.md §7](walking_clean.md#7-learnings))
- **Reward terms that pay for standing still win.** Velocity kernels and contact or tip-height imitation all pay a
  robot that doesn't move. Use progress and lift terms that are zero when standing. ([walking.md §10](walking.md))
- **Reward weights are fragile.** Paying a lot for lift (stride ×60, ref_swing ×15) taught policies to yank the foot
  up and fall; a progress term taught them to slide. Carrying the wrong set over from another run cost two rounds.
  ([walking_ankle.md §7](walking_ankle.md#7-learnings))
- **The reference sets the landing.** Penalizing touchdown speed didn't soften landings; slowing the reference's swing
  did (1.5–5× slower touchdowns). Higher steps also land more gently. ([walking_ankle.md §7](walking_ankle.md#7-learnings))
- **Check the open-loop reference first** (`scripts/ref_play.py`). It finds toe-digs, sliding and falls in minutes,
  before any training. ([walking_ankle.md §7](walking_ankle.md#7-learnings))
- **Pushes and softer rugs in training didn't help** (`c3p` crossed its feet, `c3r` no better on the soft rug).
  ([walking_clean.md §7](walking_clean.md#7-learnings))
- **Smoothness is a sim-to-real property.** Command jitter the simulator absorbs comes out as servo buzz on the robot.
  Tune the action-rate penalty for the robot, not for the simulator. ([walking_clean.md §11.3](walking_clean.md#113-full-body-c3f_t12))
- **An action low-pass filter (`act_lpf` 0.7) is the strongest tool against the shake.** It halved the legs-only
  shake in simulation and put the full body's shake at 7.4 V below the old policy's at 5 V; 0.5 was too weak for the
  legs. It is part of the policy, so the runner must apply it (it reads it from the run's config).
  ([walking_clean.md §12.4](walking_clean.md#124-learnings))
  Round 3 found a better tool, below: on the robot the filtered policies failed for other reasons.
- **Against command jitter, a Lipschitz penalty beats filters and smoothness rewards.** A gradient penalty on how fast
  the action changes with the sensor inputs (`--lcp 0.0005`, [Chen et al. 2024](https://arxiv.org/abs/2410.11825))
  cut the command jitter 4× (1.03° → 0.24-0.30° in simulation) with no filter lag, and those policies walked furthest.
  It is sharp-edged: 0.001 and 0.002 broke the policies. More joint-speed noise in training (1.0 rad/s, Open Duck uses
  2.5) helps less (0.55°). ([walking_clean.md §13.5](walking_clean.md#135-learnings))
- **Change one thing per fine-tune.** The same three changes reached in two stages gave the best full-body policy;
  all at once from the old policy, it fell 5/6. Switching straight to a wider gait made the legs march in place;
  continuing from a checkpoint that had partly adapted worked. ([walking_clean.md §13.5](walking_clean.md#135-learnings))
- **Asking for speed doesn't lengthen steps.** A higher speed command, heavier reference tracking (shorter steps) and a
  heavier speed reward (shuffling) all failed; only the per-step stride reward ×2 gave longer steps. Policies take
  about half the reference's step, so on the robot command enough speed (`--vx 0.02`-`0.03`, not 0.015).
  ([walking_clean.md §12.4](walking_clean.md#124-learnings))
- **Mirrored episodes (`mirror_aug`) even out the two feet.** ([walking_clean.md §12.4](walking_clean.md#124-learnings))
- **Train against disturbances stronger than the real ones.** Sudden pulls up to 1.5 N raised the tolerance to a
  0.5 N cable-like pull from 5/6 falls to 0-1/6; pulls up to 0.6-1 N taught little, and gentle realistic ones made the
  full body step in place. ([walking_clean.md §12.4](walking_clean.md#124-learnings))
- **New body, new fine-tune.** The legs-only walker falls in 13/16 simulated walks with the torso and head on; a
  fine-tune on the full-body model fixed it (0/16). ([walking_clean.md §10](walking_clean.md#10-full-body-torso-and-head-on-2026-11-10-evening))
- **Fine-tune with the critic restored and a lower learning rate.** A random critic plus a changed reward destroyed a
  good policy within 10 M steps. ([experiments.md, Lessons](experiments.md#lessons-process-not-just-results))
- **Put hardware limits in the reward, not only the simulator.** Clipping torque alone still produced policies 1.3×
  over rated torque. ([experiments.md, Lessons](experiments.md#lessons-process-not-just-results))
- **Match the training start to the robot's.** A harsh reset with slow servos ended most episodes early; matching it
  to the hardware made training ~3× faster. ([sim_to_real.md §3.3](sim_to_real.md#33-train-from-the-start-the-robot-actually-gets))
- **Randomization isn't free robustness.** It only helps against the failure modes it includes.
  ([experiments.md, Lessons](experiments.md#lessons-process-not-just-results))
- **A deterministic walker with IMU feedback overfits its tuning** (8 of 24 unseen walks fell) and is far behind the
  RL residual on the same reference. ([walking_clean.md §6](walking_clean.md#6-deterministic-walker-no-learning))

## 4. Modelling the robot in simulation

- **The servo firmware's motion profile was the biggest sim-to-real gap.** Every STS3215 follows a constant-acceleration
  ramp: ~400 °/s² on the 1:345 hips, ~1900 °/s² on the 1:191 knees. A 5° hip correction takes ~250 ms. Spec sheets
  say nothing about it. The fitted model matches the hips 4× better. ([servo_identification.md](servo_identification.md),
  [sim_to_real.md §3.1](sim_to_real.md#31-the-servo-firmwares-motion-profile-was-the-big-gap))
- **Acceleration, not speed, limits a quick step** (~35 rad/s², ~0.1 s lag). Check a reference's accelerations
  against the profile. ([walking.md §10](walking.md))
- **Fast hips matter more than voltage.** Recovery training plateaued on the slow hips; 1:191 hip pitches took
  simulated recovery from 0–33 % to 100 %. 7.4 V barely helps the profile, though 5 V does cost ~30 % of torque.
  ([sim_to_real.md §3.6, §3.8](sim_to_real.md#36-more-training-doesnt-fix-slow-hips-faster-hips-do))
- **Policies are fragile exactly where hardware differs.** 1° of backlash took the original stand policy from 75 % to
  6 % quiet stands; randomizing the measured servo behaviour fixed it. ([sim_to_real.md §3.2](sim_to_real.md#32-the-original-policy-was-fragile-in-exactly-the-ways-real-hardware-differs))
- **The floor model matters more than the policy.** Scaling rug softness by pad area made the training rugs nearly
  firm for big boards; the robot's falls only appeared on a softer rug. Every policy still falls on the softest
  simulated rug. ([walking.md §2, §10](walking.md), [foot_pads.md](foot_pads.md))
- **Check the servo model at the frequency where the robot misbehaves.** The fitted model matched step tests, but at
  5 Hz the real joints pass about twice as much motion with less lag, so a 4.8 Hz roll oscillation of the gyro →
  policy → roll-joint loop never appeared in training. Measure command → position gain and phase from the robot logs
  and compare with the simulator (`scripts/robot_probe.py`, [walking_clean.md §12.1](walking_clean.md#121-what-the-robot-logs-showed)).
- **Compare each joint's tracking under load, robot vs simulator.** With one foot up, the robot's standing hip roll
  sags 1.4° under the body's weight, the model's 3.5° (knees matched). The soft model hip sagged into the weight shift
  and did part of the policy's work: the robot lifts each foot with its weight 44-59 mm inside the standing board, the
  simulator 21-31 mm. With the hip rolls 2.5× stiffer (`kp_hip_roll`), the simulator reproduced that, `c3m`'s
  in-place marching at 7.4 V and `c3f`'s early falls there; training with it fixed them in simulation.
  ([walking_clean.md §13.1](walking_clean.md#131-what-the-robot-logs-showed))
- **Encoders don't see everything that bends.** The stiffer hip servo of the lesson above was right, but the hip as a
  whole gives ~5° sideways under one-leg load beyond the servo, invisible to its encoder. Policies trained without it
  shifted their weight further than the robot can stop and tipped over the standing leg; the give in the simulator
  (k ≈ 3 N·m/rad at the hip roll) reproduced every robot fall and `c3m`'s survival. Find such gaps by playing the
  robot's logged commands open-loop into the simulator and by starting the simulator from the robot's logged state
  mid-fall: if the simulator recovers where the robot didn't, the body model is wrong, not the policy.
  ([walking_clean.md §14](walking_clean.md#14-why-13s-policies-fell-sideways-the-legs-give-where-the-encoders-cant-see-2026-11-11-evening))
- **Measure the robot's sensor noise and train with at least that much.** The policies' commands followed the
  servos' quantized speed readings: that was the knee twitch. ([walking_clean.md §13.1](walking_clean.md#131-what-the-robot-logs-showed))
- **Emulate the robot's IMU filter in the loop.** The real tilt reading was off by ~5° during wobble (gyro saturation,
  accelerometer trusted during acceleration). ([README §15](../README.md#15-big-feet-for-soft-rugs))

## 5. Mechanics

- **Without an ankle, the board's length is the fore-aft support.** Pads shorter than ~100 mm fall more and balance on
  one foot for under 0.5 s; round pads fail; 150 × 50 mm boards walk on every floor. ([walking_ankle.md §7](walking_ankle.md#7-learnings))
- **Straight legs with flat feet and no ankle pitch can't lift a foot** (a singular pose). Keep a bent stance; the
  ankle fork carries a 16° angle so the shin can stay straight. ([walking_ankle.md §7](walking_ankle.md#7-learnings))
- **Flat boards fix the shin angle**: no skiing, a forced staggered stance, large knee travel to lift. A reference
  that ignores this is rejected by RL. ([walking.md §1, §10](walking.md))
- **Feet are not the whole story.** In the original study, any flat contact stopped the micro-stepping, but feet alone
  didn't fix short steps. ([README §3.4, §4](../README.md#34-control-experiments-is-footlessness-really-the-problem))

## 6. Evaluating and choosing policies

- **Pick checkpoints on the step test, not the reward.** Stepping peaks around 10 M steps; later checkpoints trade
  steps for survival. ([walking.md §10](walking.md))
- **Declare checkpoint selection before seeing test results**, and validate on unseen seeds and longer walks.
  ([experiments.md, Lessons](experiments.md#lessons-process-not-just-results), [walking_clean.md §6](walking_clean.md#6-deterministic-walker-no-learning))
- **Analyse the robot's failure before training more.** Each time, the robot logs pointed to one moment (first lift,
  touchdown, after the release), and the simulator only helped once it reproduced that moment. ([walking.md §10](walking.md))
- **Measure the start separately** (`scripts/walk_start.py`). ([walking.md §10](walking.md))
- **Replay the policy against the robot's logs.** Feeding each logged observation back through the policy reproduces
  every logged action exactly. That settles what the robot actually ran (the filter blamed for a twitch was never on)
  and gives the policy's sensitivity to each input. ([walking_clean.md §13.1](walking_clean.md#131-what-the-robot-logs-showed))
- **Score the failure modes, not just falls.** `scripts/robot_probe.py` reports command jitter (comparable with the
  robot logs), board spacing, and where the pelvis is over the standing board at lift-off and at its widest lean. The
  lift-off number matched the robot only once the hip model did. ([walking_clean.md §13.3](walking_clean.md#133-evaluation))
- **Correlation in robot logs isn't cause.** Servo buzz tracked low voltage, but load drove both; a fixed-supply
  simulation showed voltage wasn't the cause. ([walking_clean.md §11.3](walking_clean.md#113-full-body-c3f_t12))

## 7. Robot bring-up and running

- **Check every joint's direction with a small move before any sweep.** The mirrored right leg's directions are all
  opposite to the left's; the first guess for both ankles was wrong (`biped_bench.py play check_ankle`: the outer edge
  must go up). ([README §12.1](../README.md#121-bringing-up-the-right-leg), [walking_clean.md §8](walking_clean.md#8-running-these-on-the-robot))
- **A servo reused from another joint keeps its old angle limits** and may not move. `feetech_bench.py info` shows
  them; `clear-limits --id N` resets them. New servos answer at ID 1; set IDs with one servo on the bus.
- **The acceleration register (41) doesn't change the old hips' profile**; the runner writes 254. Two "identical" new
  hip-pitch servos ramped at different rates. ([sim_to_real.md §3.7, §3.8](sim_to_real.md#37-only-one-of-the-swapped-servos-is-fast))
- **The IMU zero is off by degrees**: pitch 1–2.5° (`--pitch-trim 2`), roll +1.4–3.3° on the big pads. Without ankles,
  2° of lean was past the pad's edge. ([sim_to_real.md §3.4](sim_to_real.md#34-the-balance-point-and-the-imu-zero-matter-by-single-degrees), [walking.md §9](walking.md))
- **USB/IP in WSL stalls the serial devices** (`vhci_hcd ... -104`): feetech commands hang in `write`, and the runner
  stops on "IMU silent". Re-attach with usbipd, raise `--imu-timeout 0.5`, or use native Linux USB.
- **Hold, don't drop, on a fault.** `policy_stand.py` rides through a few failed reads, then holds the stance with
  torque on (`--limp-on-fault` for the old behaviour). It logs each servo's voltage, temperature and load.
- **IMU plumbing**: the "MPU6050" is an MPU6500; hardware I2C failed, so the RP2040 runs software I2C at 100 kHz;
  read the serial stream as it arrives (big blocks added 27 ms). ([sim_to_real.md §3.8](sim_to_real.md#38-smaller-lessons), [README §13](../README.md#13-imu))
- **Supply**: 5 V sags to ~4.7 V under walking load on the full body; 7.4 V is the plan for torque, not for the buzz.
  Higher voltage makes the servos stiffer and faster, which raised the gain of the shake loop: the old full-body
  policy fell at once at 7.4 V. Round 3's smooth policies keep their steps at 7.4 V in simulation; try it after 5 V.
- **Read the robot logs before retraining.** A spectrum of the gyro and the commands showed the shake was one
  frequency (4.8 Hz) driven by the policy's roll commands, which pointed straight at the fix.
  ([walking_clean.md §12.1](walking_clean.md#121-what-the-robot-logs-showed))
- **The runner writes acceleration 254 to every servo**, but the model gives the yaw, roll and ankle joints ~400 °/s²;
  the new ankle servos honour the register. `--accel-match` sets each servo as trained. Untested on the robot.
- **Cables can pull the robot over; no policy can stop it.** On one 50 mm board the sideways tipping force at pelvis
  height is about m·g·0.025 / 0.37 ≈ 1 N, less while stepping; 0.8 N fells all but one policy in simulation. Hang the cables
  from above with slack. ([walking_clean.md §12.4](walking_clean.md#124-learnings))
- **Start the policy as it was trained.** With the runner's 1 s blend-in and the gait clock running, the first right
  step happened half-engaged and landed inward; most of the day's near-falls came at the next lift-off. The runner
  now engages ready-pose policies fully at once, and any blend holds the clock. ([walking_clean.md §13.1](walking_clean.md#131-what-the-robot-logs-showed))

## 8. Working process

- **Audit actuators from first principles before training.** It rejected a servo before any GPU time was spent.
  ([experiments.md, Lessons](experiments.md#lessons-process-not-just-results))
- **One GPU job at a time, queued, with a watchdog.** Queue scripts (`scripts/*_queue*.sh`) record every setting;
  ~20M steps take ~21 min on an RTX 4070 SUPER. Check on long jobs every ~20 min so a dead job isn't missed.
- **Keep every policy the user might test** (final checkpoint of every run is in git, whitelisted in `.gitignore`;
  `config.json` and `progress.jsonl` for every run). Don't delete policies without asking.
- **State what was not verified.** Robot claims come from trials with a hand nearby; simulation claims from held-out
  seeds. Keep the two apart.

## 9. Open questions

- Why `c3m_T12` transferred and `c6_T15` / `c5i_big` didn't (guess: least time on one foot).
- How much of the first walk is the ankle and how much the training (no ablation on the robot).
- The full-body robot marches in place on the robot but walks 36–38 cm / 30 s in simulation. Round 3's guess: the
  model's soft hip rolls did part of the weight shift (with the robot's stiffness, `c3f` takes 24-26 mm steps in
  simulation too). Untested on the robot.
- The shake: the robot logs show a 4.8 Hz roll loop through the policy; whether the action filter, `--accel-match`
  or both fix it on the robot is untested. The simulator, even with faster servos, doesn't lock into the oscillation
  the way the robot does (missing frame flex or backlash?).
- Why §12's policies leaned over the standing leg on the robot. The stiffer-hip model doesn't reproduce it (it makes
  every policy lean less), so the cause is still unknown.
- Every policy falls on the softest simulated rug (~6 mm sink).
- Whether the slow right hip-pitch servo was resolved ([sim_to_real.md §3.7](sim_to_real.md#37-only-one-of-the-swapped-servos-is-fast)).
