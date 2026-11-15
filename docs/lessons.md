# Lessons learned (start here)

What this project has learned, in one place, for whoever picks it up next: me later, or an agent. Each lesson is a
short statement plus a link to where it was learned, with the data. The detail docs keep the full record; this file
keeps the conclusions. **Add to it whenever a round ends.** Last updated 2026-11-15, after the hardware fixes that made
k5 the best full-body walker.

Status in one line: the robot (10 STS3215 servos, ankle roll, 150 × 50 mm snowboard pads, IMU on the pelvis) walks
with the legs alone for minutes (`b7_seed2`: 122 s, 99 s) and, since 2026-11-13, with the torso and head on:
`python policy_stand.py run --run h4_t05 --enable --vx 0.015 --act-lpf 0.6` walked 188 s (299 steps), 74 s and 76 s.
Its remaining faults: bouts of twitching against the frame's 5 Hz ring, and falls after a minute or so. The full
detail: [walking_clean.md §18-22](walking_clean.md#22-round-7-on-the-robot-the-ring-is-tamed-the-steps-tip-forward-2026-11-13-evening).
Rounds 8-9 tried to make it faster and calmer; nothing beat it on the robot yet, but they showed why it works
([walking_clean.md §26](walking_clean.md#26-round-9-on-the-robot-the-new-gait-tips-sideways-the-gyro-penalty-transfers-and-why-h4-works-2026-11-14-1104-1132)).
Round 10 on the robot ([§28](walking_clean.md#28-round-10-on-the-robot-high-steps-transfer-open-loop-removes-the-twitch-and-why-the-steps-tip-2026-11-14-1643-1820)):
`k5_high2` (h4 on a reference that lifts 2.1x higher) takes the large, clean steps the next milestone needs but tips
sideways; the open-loop policies have practically no twitch. Next (round 11): k5's steps with the open-loop
policies' indifference to the sideways sway, a live slow pitch channel, and lift-off timed to the roll.

**Best full-body walker since 2026-11-15: `python policy_stand.py run --run k5_high2 --enable --vx 0.015 --act-lpf 0.7
--video`**: median 25.7 s and 30 steps over 17 trials (max 32.9 s / 42 steps), large clean steps, much faster than
h4 per step; slightly less reliable. Remaining fault: in some trials the left step grows wider and the robot drifts
left ([walking_clean.md §31](walking_clean.md)).

## Hardware (2026-11-15)

- **Check the screws when the shake comes back.** Loose hip-yaw servo screws: body shake 17.7 deg/s; tightened
  14.5, and k5 stopped falling in the first steps. No training change did as much for the vibration.
- **Check the zeros standing on the floor.** A few degrees of zero error on one leg tilted the stance and sent every
  fall to that side; the policies can't learn their way out of it. `biped_bench.py trim` re-zeroes standing.
- **Servo EEPROM writes: clear the serial input buffer afterwards, and never exit with a joint limp.** A late reply
  shifted every later read, so correct zeros were reported FAILED, and the tool dropped the leg.

## Video and round 12 (2026-11-14)

- **Record the robot.** Encoder-based foot heights are wrong by several mm under load (invisible give); the video
  showed the "early lift" was a lean and that the falls start in the weight shift. `--video`, `video_frames.py`.
- **A gate that holds the clock must never hold with a foot up or a fall under way**: it freezes the reference with
  the foot in the air. The runner's gate guard (default on) skips those holds.
- **Check for a sideways bias in the robot before training robustness to it.** All falls went left; the IMU zero
  was fine; the left hip carried twice the load. A wide symmetric mass range (+-2 cm) made k5 worse in 12-16M.
- **Large new randomization ranges degrade a fine-tuned high-step policy** (round 11 hips, round 12 mass): widen
  gradually, or measure and train around the real value.

## Round 11 (2026-11-14, simulation)

- **The fitted simulator is too stiff sideways for high steps.** A hip roll with twice the fitted give reproduces k5's
  robot lift-offs (still rolling outward at ~20°/s) and its sideways falls, and ranks h4 / k5 as the robot does; screen
  high-step policies on `eval6 --sim loose`. ([walking_clean.md §29](walking_clean.md))
- **Never low-pass the pitch rate the policy sees below ~3 Hz**: 1 Hz alone made k5 fall forward 6/6. Delayed rate
  feedback adds energy.
- **One change at a time from a working policy.** Sideways open loop + high steps + loose hips together gave 2.5 s
  episodes that didn't improve in 20-24M; loose hips alone needs well over 16M.
- **The roll gate (`--gate-roll`) is harmless at run time**; the runner now logs `phase`, `gate_held`, `roll_pred`.

## Full body: what round 10 taught on the robot (2026-11-14)

- **High steps are the direction for mobility.** A higher reference gait transfers (k5 lifts ~2x h4 on the robot):
  more progress per step, no non-steps, no toes caught in the rug. Their cost is sideways risk, below.
- **A sideways fall is a lift-off taken while the body still rolls outward.** All 22 sideways falls came mid-swing,
  over the stance foot; each of those steps lifted at +5-7° toward the stance side, still rolling there at
  14-28°/s. Normal h4 steps lift at +0.5°; k5 lifts at +4.6° and 21°/s every step, near the edge, and stands on one
  leg longer. Fix the timing of the lift, not the height. ([§28.1](walking_clean.md))
- **The sideways sway is safe; fighting it is not.** The open-loop policies lift mid-sway (0° roll), let the
  momentum carry ~2° over the stance foot and back (the "strafe"), and never fell sideways. The closed-loop policies
  fight the sway, which is the twitch, and lift late.
- **Open loop within a step removes the twitch** (swing gyro RMS 12-16°/s against 20-25), **but the fore-aft balance
  must stay closed loop.** Holding the pitch rate up to 1.2 s, even with gravity live (o3), lets a fore-aft rock grow
  (10 → 16-22° peak-to-peak per 2 s) until it falls. Hold the sideways and joint inputs; keep pitch live and
  low-passed below the ring.
- **Held inputs need a clean start.** Frozen at the engage frame, with the settle gate holding the clock off the
  first sample, the o policies were blind through the engage lurch and fell on their faces.
- **Engage into the policy's own pose.** Every policy lurched -5 → +10-12° forward within 0.75 s of engaging from the
  reference's ready pose. `--ready-act` (blend into the pose the policy holds, clock at 0) removes it; it is the
  runner's default since 2026-11-14.
- **No gyro gradient penalty.** It cuts the twitch by cutting all gyro gain, the pitch damping included: k2, k6 (round
  10) and d6 (round 9) fell forward, d6b tipped sideways. Cut the ring with the run-time filter or by not showing it
  to the policy.
- **Check the settle gate.** Lift-offs passed at 21-28°/s through a 17°/s gate; either it hit its 1 s cap or the
  policy lifts ahead of the clock.

## Full body: what rounds 8-9 taught (2026-11-13 / 14)

- **Train without a ring rather than with a wrong one.** A policy trained against a simulated ring learns a
  counter-move timed to that ring; on a body that rings differently the same move feeds the ring. On the no-ring
  simulator, the shake grew with the training ring's strength (7 → 8 → 20 → 33°/s at 0, ½, 1, 1.5× the round-7 ring),
  and no ring-trained policy did better on the robot. ([walking_clean.md §25.4, §26](walking_clean.md))
- **Train without the action filter, and apply it at run time.** With the filter in training the smoothness penalties
  only see the filtered action, so the policy pushes harder behind it: the hip's gyro gain doubled every time (s1, s3,
  d5, d11), and those twitched worst on the robot.
- **A gyro-only gradient penalty cut the twitch on the robot, but not usefully.** `--lcp 0.01 --lcp-idx 3:6` (the
  Lipschitz penalty of [Chen et al. 2024](https://arxiv.org/abs/2410.11825), on the gyro inputs only) halved the 4-8
  Hz commands against h4's; 0.05, or no gyro at all, removed the fast balance too: those fell forward or twisted
  within 3 s. Round 10 retired it (k2 and k6 fell forward; see above).
- **More reward pressure means more twitch.** Doubling the progress and swing pay sped the policy up and made it shake
  ~30% more. h4's light weights are part of why it is calm.
- **The gait must let the landing settle.** Cutting the double support (release 0.25 → 0.15 s, the settle gate's
  hold 1.0 → 0.2 s) made every policy lift the next foot while the body still rolled from the last landing, and tip
  over the stance foot at the third step. The simulator doesn't charge for this; keep the wait after landing and
  shorten other phases if faster stepping is wanted.
- **Rewards on the symptom teach avoidance.** A landing-speed penalty taught hesitation; fake gyro vibration taught
  nothing to a policy with no memory.
- **Make the step target follow the command.** h4 walks ~1.2 cm/s at any command because its fixed 5 cm step target
  paid standing about as well as walking; `stride_cmd` (target = speed command / step rate) fixed the speed in
  simulation.

- **A higher reference gait teaches a higher step; a clearance penalty doesn't.** h4 lifts its feet a quarter of
  its reference's height (6.7 mm median) with the clearance penalty on; references lifting 1.4× / 2.1× higher raised
  the real lift 2.2× / 5×. ([walking_clean.md §27](walking_clean.md))
- **Open loop within a step (sensors held between lift-off and touchdown) lowers the shake but costs the slow
  balance**: the least shake on the hard ring, but half the posture answer and a forward lean. It needs a live slow
  channel or longer training. (§27)
- **The step target following the command doesn't buy speed by itself**; the centre of mass range and the higher
  steps did. (§27)

### Why h4_t05 works, and what to improve

| h4 has | why it matters |
|---|---|
| No ring in training | its gyro answer is generic, so the run-time filter can cut the loop where the robot rings |
| Light reward pressure (tilt -0.05, progress 1.5, pose σ 0.1) | nothing pushes it to answer every disturbance |
| No filter in training, `--act-lpf 0.6` at run time | the smoothness penalties see its raw actions; the filter removes only the 5 Hz part |
| A gait that waits for calm (c_T12, 0.25 s release, gate up to 1 s) | the landing roll dies out before the next weight shift |
| A strong slow posture answer (-0.57° knee - hip per degree of lean) | it catches slow leans; policies that lost it fell forward |

| weakness | fix to try |
|---|---|
| ~1.2 cm/s at any command (fixed 5 cm step target) | `stride_cmd` at h4's pay level |
| Twitch bouts (gyro gain 0.047 / 0.096) | gyro-only LCP 0.01 (round 10: cuts the pitch damping too; out) → hold the sideways inputs within a step |
| Leans 4.5° forward (model's centre of mass 1-2 cm back) | `dr_com_x` [-0.005, 0.025] |
| Toes catch | a reference gait that lifts higher (a clearance penalty alone doesn't teach lifting); round 10: works (k5) |
| Jolt at engage | blend the first actions in; `--ready-act`, default since round 10 |
| No memory to tell the ring from a real rotation | observation history (experimental) |

## Full body: what rounds 5-7 taught (2026-11-12 / 13)

- **The full body's shake was a feedback loop, not a mechanical fault.** The hollow frame rings at 4-5 Hz (tap test,
  ζ ~0.15), the pelvis gyro sees it, the policy answers it at the hip pitches and knees, ~80 ms late, and the servos
  ring the frame again. Replaying the logged sensor readings through the policy reproduced its commands exactly;
  smoothing the gyro alone removed ~40% of the shaking commands. The legs-only robot doesn't ring, so the leg policies
  never started the loop. ([walking_clean.md §20](walking_clean.md#20-round-6-on-the-robot-the-shake-is-a-feedback-loop-2026-11-13-morning))
- **Measure the body's response to small commands, not only taps.** A wiggle exercise (small sines at 1-7 Hz,
  `exercises.py wiggle`) showed the real body pitching 8-10× more per degree of hip-pitch command at 4-5 Hz than the
  simulator's; taps had shown the ring but not that. ([measuring.md E8](measuring.md))
- **Gain round the loop decides the shake.** Policy gain (commanded degrees per °/s of gyro) × body gain: ~2 on the
  robot, ~0.8 in the simulator. Full-body policies had learned 4-5× the leg policies' pitch-rate gain in a simulator
  where it cost nothing.
- **Training in a simulator that rings is not enough on its own.** With the ring fitted in (a damped oscillator driven
  by the servo targets, torque on the pelvis), 30-70M steps of fine-tuning, observation history or a gyro low-pass did
  not lower the chatter. An action low-pass (0.6) did, by construction, in training and at run time.
- **A run-time action filter on a policy trained without it beat the policies trained with it.** The round-7
  filtered policies tamed the ring on the robot but leaned 7-12° forward and caught their toes, a lean their
  simulator didn't show; h4 + `--act-lpf 0.6` kept its round-6 steps and lean. Change one thing at a time when a
  policy already walks.
- **The robot stands at its backward tip limit** (+5 to +7° back with the boards flat, back tip at 7.5°), so every
  policy snaps the body forward when it engages; under load the knees don't move for ±1° commands.

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
- The shake (answered 2026-11-13): a loop through the frame's 5 Hz ring and the policy's gyro gain; the action
  low-pass breaks it ([walking_clean.md §20-22](walking_clean.md)). Open: why the round-7 policies lean 7-12° forward
  on the robot but not in the simulator (the ring, the 60 ms dead time or the shallower standing lean in training).
- Why §12's policies leaned over the standing leg on the robot. The stiffer-hip model doesn't reproduce it (it makes
  every policy lean less), so the cause is still unknown.
- Every policy falls on the softest simulated rug (~6 mm sink).
- Whether the slow right hip-pitch servo was resolved ([sim_to_real.md §3.7](sim_to_real.md#37-only-one-of-the-swapped-servos-is-fast)).
