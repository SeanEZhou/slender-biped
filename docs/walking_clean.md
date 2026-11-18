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
| **`h4_t05` + `--act-lpf 0.6`** | full body, round 6 (§19); action low-pass at run time | **the best full-body walker until round 10: 188 s (299 steps, stopped standing), 74 s (115, fell left), 76 s (109, ended at 7° roll)** on 2026-11-13 evening; pitched 4.5° forward. With `--act-lpf 0.7` (22:16): 147 s (251 steps) and 138 s (227), both ended standing; at `--vx 0.02` 84 s, 67 s and 32 s (fell right in a twitch bout) (§24). It sometimes starts fighting the 5 Hz ring and twitches when there's no need. `python policy_stand.py run --run h4_t05 --enable --vx 0.015 --act-lpf 0.6` | §20, §22 |
| **`k5_high2` + `--act-lpf 0.7`** | full body, round 10 (§27): h4 on the higher-stepping reference c_T12HH | **the best full-body walker since 2026-11-15** (after the hip-yaw screws were tightened and the left hip pitch re-zeroed, §31): 17 trials, median 25.7 s up and 30 steps (max 32.9 s, 42 steps), 7 falls (4 left, 3 right). Large, clean steps (~2x h4's lift): much faster progress per step than h4, slightly less reliable. Some trials drift left: the left step lands wider and shorter (§31). `python policy_stand.py run --run k5_high2 --enable --vx 0.015 --act-lpf 0.7 --video` (`--no-gate-guard` or not: similar) | §27, §28, §31 |
| **`l9_s18` + `--act-lpf 0.7`** | full body, round 17 (§37): c_S9mid18, a 0.6 s soft swing with the weight held over the stance foot (shift 0.18), from k5 via m2, l4, l7 | **the softest, quietest steps so far** (2026-11-16): 41 s (33 steps, fell back) and 48 s (37 steps, stopped standing); peak landing jolt 30 % below k5 / m2. The swing ends skimming the rug, so steps feel like small trips (§37.1). `python policy_stand.py run --run l9_s18 --enable --vx 0.015 --act-lpf 0.7 --video` | §37 |
| **`p1_s13` + `--act-lpf 0.7`** | full body, round 19 (§39): c_P1s13, n15's swing with the weight shift cut from 0.18 to 0.13, from n15 | 2026-11-16: 2 × 50 s walks, 5 falls in 152 steps. Even smoother than l9 (body roll 9° per cycle against 13°; the gate pauses 2.5 % of the time against 4.3 %) but slower: 34 mm strides at 14 mm/s against l9's 42 mm / 17 mm/s. `python policy_stand.py run --run p1_s13 --enable --vx 0.015 --act-lpf 0.7 --video` | §39.1 |
| **`p4_gate` + `--vx 0.03 --act-lpf 0.7`** | full body, round 19 (§39): p1 with the settle gate's lean limits relaxed to 5 / 8° (from 3 / 6°) | **the most stable, smooth and continuous gait so far**. At `--vx 0.015` (2026-11-16): 115 steps without a fall, 3 of 4 trials stopped upright after 45 s, but the slowest (34 mm strides, 14 mm/s). At `--vx 0.03` (2026-11-17): 42 mm strides at 17 mm/s, 78 steps without a fall. A small balance loss at lift-off can escalate to a fall within one step (§41). `python policy_stand.py run --run p4_gate --enable --vx 0.03 --act-lpf 0.7 --video` | §39.1, §40.1, §41 |

All three round-4 policies walk from the first step; their falls now come after 15-28 s and are mostly backward (§16).
Tried on the robot and failed (the bar for this table: 10+ steps in every trial): `fb2_lean`, `fb4_soft` (full body: step, but
5-24 s, every trial falls sideways, a 4.5-5 Hz shake on most steps, §18); `c6_T15` (stepped on its own foot), `c5i_big` (tipped every other step) (§11);
`h9_w6kl_L`, `h4_w6k_L`, `h13_kl_L`, `h3_k_L` (fell sideways in the first steps, §14). Every walker before the ankle
roll fell on the first or second step ([walking.md](walking.md)). Untested on the robot: the other §12-13, §15 and §17
runs. With the cables no longer pulling (§18), all of round 5's leg policies and the s policies walk for a minute or
more; `c3m` still falls sideways within 10-20 s. The legs are now steadier than the full body, which is the gap to close
next (§18.4). Round 6 (§19-20): `h2_old`, `h5_h2hard`, `h4_t05` all fell in 5-24 s on the robot, shaken over by a gyro feedback loop at 4.5 Hz. Round 7 (§21-22): the policies trained with the action filter stopped fighting the ring but walk 7-12° forward and fall on their faces; **`h4_t05 --act-lpf 0.6` is the current best full-body walker.** Round 8 (§23, simulation): the round-7 lean matches a centre of mass 1-2 cm further forward than the model's; `s3_swing` (h4 + the filter trained in + the ring + more swing pay) is the next to try. Round 8 on the robot (§24): s3 twitched worst, s2 stepped in place, h4 + 0.7 as good as 0.6. Round 9 (§25, simulation): a 1.0 s continuous gait with 1.5x steps; `d1b_ignore` walks 4.7x h4's speed in simulation; training on a stronger ring makes policies shake a body that rings differently; the gyro-only gradient penalty halves the gyro gain (`d6b_glcp01`). Fourteen policies to test. Round 9 on the robot (§26): every c_T10L policy tips sideways at the third lift-off, because the gait no longer waits for the landing roll to die out; the gyro-only penalty (`d6b_glcp01`) is the first training change that cut the twitch on the robot; h4 + filter is still the best. Round 10 (§27, simulation): single changes from h4; a higher reference gait doubles to quintuples the real foot lift, the centre of mass range gives the best ring screen, the gyro-only penalty halves the gain, and holding the sensors between steps lowers the shake but costs the slow balance. Round 10 on the robot (§28): `k5_high2` steps the way the next milestone needs but tips sideways, because it lifts off while still rolling outward; the open-loop policies have practically no twitch and never fall sideways, but blind pitch lets a fore-aft rock build; the gyro penalty is out; `--ready-act` (now default) removes the engage lurch. Round 11 (§29, simulation): a hip roll with twice the fitted give reproduces k5's robot falls; a 1 Hz pitch-rate low-pass breaks the fore-aft balance; k5 + sideways open loop + roll gate + loose hips at once did not train in 20-24M. §30: a camera on the robot; the falls now all go left (a sideways bias in the robot); wide sideways mass ranges made k5 worse. §31: the hip-yaw screws tightened and the left hip pitch re-zeroed; **`k5_high2 --act-lpf 0.7` is the best full-body walker** (median 25.7 s, 30 steps). Round 13 (§32): shoves and penalties taught no recovery step (planted balance only, and on the robot the policies did almost nothing while falling); next, recovery-step references chosen by the capture point.

## 0. How the walking policies work: a reference gait plus corrections, on a clock that can wait

Every policy in the table above works the same way: `c3m_T12`, `c3f_T12`, the s and b leg policies, `h4_t05` and
`k5_high2`. Their run configs all have `ref_residual` true, a c_T12-family table, `action_scale` 0.25, the settle gate
at u = 0.2083 / 0.7083 holding up to 1.0 s, and `heading_cmd`.

**The reference is the anchor; the policy only corrects it.** Each servo target is

    q_target = reference(vx command, clock phase) + 0.25 rad x action,   action in [-1, 1]

clipped to the joint limits. The policy never outputs raw servo positions: at most +-14 deg per joint around the
reference's pose. The step the robot takes is mostly the reference's step, bent by the policy to keep balance.
A policy is therefore meaningless without its table:
- The table is committed in `models/ref/<ref_table>.npz` (k5: `c_T12HH`; the others: `c_T12`), built by
  [ref_gait.py](../slender_biped/ref_gait.py) (§4: weight shift by the hip and ankle rolls, flat lift, settle).
- The run's `config.json` names it, and `policy_stand.py` loads it from there.

**Picking the reference frame: indexed by the clock, interpolated.** The table is 5 forward speeds (0-4 cm/s) x 256
phases x 10 joints, plus which board is down at each phase (`stance`). Each 20 ms tick:
1. The clock advances by 2 pi x 0.4167 Hz x 20 ms; one cycle is 2.4 s, a left and a right step.
2. `ref_lookup` (policy_stand.py; `ref_gait.lookup` in training) blends linearly between the two nearest phase bins
   and the two nearest speed rows.

c_T12HH's schedule over one cycle (cycle fraction u):

| u | both boards down | left board up (right stands) | both down | right board up |
|---|---|---|---|---|
| from | 0 | 0.208 (lift-off) | 0.396 (touchdown) | 0.708 (lift-off) |
| to | 0.208 | 0.396 | 0.708 | 0.896, then both down to 1 |

**The policy's inputs (k5: 41):**
- gravity direction (3) and gyro (3) from the IMU;
- 10 joint angles, 10 joint speeds x 0.1, and its previous action (10);
- the command (3): forward speed vx, sideways vy (always 0 in training, so meaningless) and the turning rate wz;
- the clock as sin / cos of the phase (2).

The clock tells the policy which part of the reference it is correcting, so it doesn't have to work the timing out
from the body's state. Stepping becomes "follow the schedule and fix it", not a reaction to be discovered.

**The command:**
- **vx** picks the stride row, and training pays progress relative to it.
- **vx = 0 is standing:** at zero speed the reference is the static stance pose at every phase. The stepping
  rewards (`ref_swing`, `stride`, `progress`) switch off, so a walker at zero command stands and does not step,
  even though its clock keeps running. This is why k5 never steps when shoved standing (round 13).
- **wz holds the heading:**
  - The runner integrates the gyro's rate about the vertical (gravity from the IMU) since engage and commands
    `wz = clip(-heading_gain x yaw, +-0.3 rad/s)`; `--heading-gain`, default 1, as in training.
  - The reference leaves the hip yaw free (`ref_yaw_weight` 0), and training paid for matching wz and holding the
    heading, so the policy steers with the hip yaws.
  - Whether the robot's straight walking needs it is untested (`--heading-gain 0` would show it).

**The settle gate: the clock waits for the body.**
- At each lift-off phase (u = 0.2083 and 0.7083) the clock stops while the body still turns faster than 0.3 rad/s
  (17 deg/s, roll and pitch gyro), for at most 1.0 s.
- While it waits, the reference holds the pose just before the lift: the weight already shifted over the stance
  board, both boards down. Not the neutral stance. The policy sees the clock stopped too, so it knows it is
  waiting.
- When the landing roll has died out (or after 1 s) the clock moves on, and the foot comes up.
- Without the gate the clock runs at a fixed rate, and the reference lifts the foot on schedule whether the body has
  settled or not. The policy can resist only within its +-14 deg.
- Gate-aware policies were all trained with it, in simulation as on the robot (`settle_gate_u`, `settle_gate_w`,
  `settle_gate_max` in the config). The runner reads the same settings.
- Round 9 showed why the wait matters: shortening it made every policy lift while still rolling, and tip at the third
  step (§26).

**The gate guard (runner only, default on since 2026-11-14).** Waiting only helps while the body swings back toward
centre. The runner skips the hold when:
- the roll (from the engage attitude) is past 4 deg, still growing, and the roll rate is above 10 deg/s;
- or a foot is already up (knees differ by more than 12 deg).

Frozen in those states, the reference would leave the foot hanging while the robot tips toward it, which is what the
2026-11-14 video showed (§30). `--no-gate-guard` runs the gate as trained. For k5 the two give similar results
(§31): k5's landings usually settle within the hold anyway.

**The engage.** The runner blends into the reference's zero-speed pose plus the action the policy holds there
(`--ready-act`, default since round 10), with the clock at 0. Then the clock starts and vx ramps up at `cmd_vx_slew`.

**Where it came from:** see the history section that follows.

**What this means for new skills.** Rewards alone never taught stepping (§3: penalties without imitation made the
policy step less and slide). Round 13's shove training without a reference produced no recovery step either. A new
motion such as a recovery step should get the same structure: a reference table indexed by its own command (e.g. the
step target) and a clock that starts when it is triggered, imitated with large `ref_q` / `ref_contact` weights, with
the penalties shaping it.

## History: how walking came to work (to 2026-11-15)

Each stage kept what the one before it got right.

1. **No reference: speed and contact rewards, and a skitter** (rounds 1-3, 2026-10-23 to 10-29,
   [experiments.md](experiments.md) E9-E45; ball-tip and small-pad legs, no boards, no ankle).
   - Actions around a fixed standing pose (stance + 0.35 x action).
   - Rewards:
     - speed and heading tracking, upright and height;
     - a gait clock saying only which foot should be down (`contact_phase`, `swing_height`, `feet_air_time`);
     - hardware penalties, and the fall penalty.
   - Every walker skittered in simulation: 5-7 touchdowns a second, 45-60 ms in the air, 1-3 cm per step, ~5 mm of
     clearance. Contact timing alone let the policy satisfy the clock with shuffles.
2. **The reference-gait architecture, but too weak to carry a step to the robot** (round 4, 2026-10-29/30, E46-E54;
   later the 150 mm boards).
   - IK reference tables, actions as corrections around the reference (scale 0.25), and the imitation terms `ref_q`,
     `ref_tip`, `ref_contact`.
   - From scratch it took real steps at once (220 ms in the air, 22 mm clearance). But it marched in place, and
     imitation couldn't reshape an already converged skitter.
   - On the boards (task `shift_b150`, 2026-11-09): corrections on a shift-and-step reference and the first settle
     gate, with weak imitation (`ref_q` 1, `ref_contact` 0) and the pay mostly for progress and lift.
   - Without an ankle there was no one-foot stance at all: every walker fell on the first or second step on the
     robot ([walking.md](walking.md)).
3. **`c3m_T12`: the reward tuned, the ankle roll added: the legs walk** (2026-11-10, §3, §11).
   - **The ankle roll servo:** a real one-foot stance. The same RL walker went from 6 steps in a quarter of the
     walks to 16 of 16.
   - **A continuous reference** `c_T12`: 1.2 s per step, weight shift and swing overlapping, boards lifted flat.
   - **Imitation as the anchor:** `ref_q` 1 -> 3 and `ref_contact` 0 -> 3 (each board up or down when the reference
     says).
   - **Then the clean-step penalties** (scuff, board tilt, roll rate, jerk). The same penalties without the anchor
     made the policy step half as often and slide (`c1_b150`).
   - The first walk on the robot (legs and pelvis). Later leg policies (`s10_robust`, `b7_seed2`: 122 s) added the
     measured hip give and the backward lean to the simulator.
4. **`h4_t05`: the first full-body walker** (round 6, 2026-11-13, §19-22). Two breakthroughs.
   - **Training on the measured body.** Re-weighed masses (torso and head 154 mm above the hips), the give as a spring
     behind each servo (`flex_series`), no cable pull. Only a *light* tilt penalty (-0.05 past 4.6 deg), so that
     stepping stayed cheap. h4 had the smoothest commands of the round.
   - **The run-time action filter.** On the robot every full-body policy shook, and the shake turned out to be a
     feedback loop: the hollow frame rings at 4-5 Hz, the pelvis gyro sees it, the policy answers it ~80 ms late,
     and the servos ring the frame again (§20). `--act-lpf 0.6`, applied only at run time to a policy trained without
     it, broke the loop: 188 s (299 steps), 74 s and 76 s (§22).
   - Filtering in training instead made the policies lean forward and fall on their faces (round 7).
5. **`k5_high2`: the faster full-body walker** (round 10, 2026-11-14, §27-28, §31).
   - **The gait:** h4 fine-tuned 16M on its own recipe with the higher-stepping reference `c_T12HH` (lift knee =
     hip 0.75 rad, against 0.45): ~2x h4's real lift. Far more progress per step, no toes caught in the rug.
   - **Kept from h4:** no ring in training, no filter in training (`--act-lpf 0.7` at run time), light reward
     pressure, the settle gate's 1 s wait. Round 10 also showed what to leave out: the gyro gradient penalty (k2, k6
     fell forward).
   - **Runner changes that made it usable:**
     - `--ready-act` (engage into the pose the policy holds, default since round 10): removed the -5 -> +10 deg forward
       lurch at engage;
     - the settle gate guard (round 12): no clock hold with a foot up or a fall under way.
   - **The hardware fixes:** the loose hip-yaw servo screws tightened (shake 17.7 -> 14.5 deg/s) and the left hip
     pitch re-zeroed. Before them, k5 had a median 7.8 s and 4 steps with falls mostly to the left. After: a median
     25.7 s and 30 steps (max 33 s / 42), falls split left / right (§31).
   - No training change did as much for k5 as the screws and the zero.

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

First results, 2026-11-12 night ([measuring.md E7](measuring.md#e7-tap-test-how-the-body-rings-full-body-2026-11-12-night)).
Tapped while holding its stance, the full body:
- rings at 4.8-6.0 Hz in roll and pitch, with a damping ratio of about 0.15. That is the walking shake's frequency,
  and the simulator has no such mode. The joint encoders barely move (0.2-0.6°).
- turns 3-4° in yaw on a twist, which dies out quickly (ζ 0.2-0.4).
- stands 6-7.5° back at rest, against 2.5-3.7° for the legs alone, so a tap from the front rocks it onto its heels.

By hand, the hip-pitch and hip-yaw servos give too (about 1° at hip pitch). The encoders see half of that or less.

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

## 19. Round 6: the full body on measured masses (2026-11-13 night, simulation)

Trained overnight on §18's measurements (measuring.md E0-E2, E6, E7). Notes and raw results: `results/robot6/`; queues
`scripts/robot6_queue{1..8}.sh`; screen `scripts/eval6.py`.

### 19.1 What changed in the simulator

- **Masses** (`scripts/make_mass_model.py`): `hw_ank3_b150_m2_10dof`, the legs, pelvis and hip plates at 0.9 kg (model
  was 1.205) and the torso and head at 0.3 kg (was 0.243), now 154 mm above the hips. `hw_ank3_b150_legs_pelvis_m2_10dof`
  is the legs alone at 0.9 kg.
- **The give as a spring behind the servo** (`flex_series`, both environments). The servo damps its own shaft, so the
  body can now ring on the give. The give can also be stiff at small loads and soft past a torque (`hidden_flex`
  entries `(joint, k, tau0, k2)`), at any joint.
  - A first version was numerically unstable, and the old policies fell within 0.5 s. Now the shaft is solved
    implicitly (`series_give`).
  - Setting used: hip roll and ankle roll k 10 (2.5 past 0.15 N·m), knee and hip pitch 15, hip yaw 10, randomized
    ×0.5-1.5, time constant 3-8 ms.
  - Hip-roll give alone barely lets the pelvis roll with both feet flat: the legs and pelvis lock as a four-bar.
    The robot's roll needs give at the ankles and knees too.
- **No cable pull**; knee offset to -0.09 rad (-0.13 in the hard runs) for the full body's deeper back lean.
- **`tilt_lim` reward:** a penalty on roll and backward pitch (forward counts half) past a threshold. Leaning back and
  sideways tips the robot over a board corner.
- **Screen:**
  - fitted: the series give above, knee -0.06;
  - hard: give ×0.6, knee -0.09, slower give;
  - old: round 5's give.

  fb2_lean, which fell on the robot, passes fitted (0/9) and fails hard (8/9, 6 s). The simulator still shakes less than
  the robot (4-8 Hz 9-13°/s against 24-31) and does not tip at E6's 3-5° of shift.

### 19.2 Results (9 × 30 s at 0.015 m/s)

| run | from, change | fitted: falls, cm / 30 s, step L / R mm, \|roll\| p95 | hard: falls | 4-8 Hz commands |
|---|---|---|---|---|
| `fb2_lean` | round 5 | 0/9, 43, 14 / 16, 7.4° | 8/9 | 3.3% |
| `g1_mass` - `g6_step` | tilt past 3.4°, -1 | 0-3/9, -15 to +8 cm: standing, shuffling | 0-4/9 | 5-11% |
| **`h2_old`** | fb2 → masses, round-5 give, tilt past 4.6°, -0.2 | **0/9, 32, 12 / 12, 5.5°** | 7/9 | 4.6% |
| `h3_notilt` | series give, no tilt penalty | 3/9 (left), 38, 17 / 17, 6.9° | 9/9 | 2.5% |
| `h4_t05` | series give, tilt -0.05 | 0/9, 32, 10 / 15, 5.9° | 8/9 | 2.3% |
| **`h5_h2hard`** | h2 + knee to -0.13, give ×0.4-1.5 | 1/9, 15, 10 / 10, 5.3° | **3/9** | 4.8% |
| `h6_h2s1` | h2's recipe, seed 1 | 0/9, 23, 12 / 9, 5.8° | 6/9 | 4.4% |
| `h7_h5step` | h5 + forward pay | 0/9, 11, 6 / 10, 5.4° | 2/9 | 6.6% |
| `b7_seed2` (legs, 0.9 kg model) | round 5 | 0/9, 63, 25 / 24, 7.2° | 4/9 | 3.0% |
| `l1_mass` (legs) | b7 + series give, tilt -0.2 | 0/9, -6, standing | 1/9 | 3.2% |

At 0.02 m/s: `h2_old` 0/9, 30 cm; `h4_t05` 0/9, 27 cm; `fb2_lean` 2/9.

### 19.3 Learnings

1. **A tilt penalty strong enough to matter stops the stepping.** Past 3.4° at -1, it cost ~1600 per episode against
   ~200 for progress, and every policy learned to stand and shuffle (`g1`-`g6`). Past 4.6° at -0.2 it keeps fb2's
   steps and cuts the roll (p95 7.4 → 5.5°). Without it (`h3`), the steps get longer and the robot falls sideways.
2. **With the series give, policies drift toward 4-8 Hz chatter** (5-11% of command power) unless the tilt penalty is
   light (`h4`: 2.3%).
3. **Harder training margins (`h5`) buy robustness and cost progress:** 3/9 on the hard sim against 7-8, at half the
   distance. More forward pay on top (`h7`) didn't bring the steps back.
4. **The recipe repeats** (`h6`, seed 1: 0/9, 23 cm, hard 6/9), unlike round 5's 2-of-3 seeds.

### 19.4 What to try on the robot

Full body, `--vx 0.015`, cables slack. Watch three things: whether it still falls sideways at the step, the shake,
and whether it stops leaning back.
1. **`h2_old`**: fb2's steps on the right masses, rocking less.
2. **`h5_h2hard`**: if h2 still tips; shorter steps, the most margin.
3. `h4_t05`: the smoothest commands, if the shake is what tips it.

If they all still fall within a few steps, the simulator's missing piece is the E6 tip at 3-5° of shift. The
next model step is then softer give at the ankles under load, fitted to E6 directly with `scripts/tap_sim.py`.

## 20. Round 6 on the robot: the shake is a feedback loop (2026-11-13 morning)

Seven full-body trials at `--vx 0.015`, cables slack (10:08-10:12, logs `stand_h*_20261113_10*.csv`). None walked.
Times include the 2 s blend into the ready pose; the policy engages at 2.0 s.

| policy | trials (s) | fell |
|---|---|---|
| `h2_old` | 13.6, 23.9, 5.7 | left, right, left; the only one to take a few (dangerous) steps |
| `h5_h2hard` | 7.2, 5.1 | right, back |
| `h4_t05` | 12.1, 7.8 | forward, forward |

The user's observation: the swaying now catches the robot well; what kills it is the shake, which starts the moment
the policy engages, before the first step.

What the logs show:

1. **The shake starts on the first tick.** The policy engages with a 0.5-0.7 jump in action (6° at a hip pitch, the
   same jump the simulator starts with). Within 0.2 s the yaw rate swings ±60-90°/s, roll and pitch ±30-75°/s, at
   4.3-5 Hz, the frame's tap-test ring (measuring.md E7). It doesn't build up over steps; it is there at full size
   from the first half second.
2. **It is the same shake as fb2 / fb4 (§18), not worse in the gyro:** 3.5-8 Hz gyro 17-30°/s against fb2's 24-31;
   the policies put 7-19% of their command power at 4-8 Hz (simulator: 2-5%). b7 on the legs-only robot: 5-7°/s, 1%.
3. **The policies feed the gyro straight back into the knees.** Replaying each log's observations through its policy
   reproduces the logged actions (to 4e-4). Low-passing one input at 2.5 Hz removes the shake's share of the actions
   by: gyro 34-41%, joint speeds 8-16%, gravity, positions or last action ≤ 4%; all inputs together 83-88%. For b7
   the gyro removes 5% (the joint speeds 47%).
4. **The full-body policies' gyro gain is 4-5× the legs policies' in pitch.** Commanded joint angle per pitch rate,
   rms over the walk, all joints (deg per deg/s): `h2` / `h4` / `h5` 0.14, `fb2` 0.10, `fb4` 0.11, `b7` 0.03,
   `s8` 0.03; in roll 0.17 against 0.11-0.15. Mostly the knees (0.07-0.10 each). The round-6 policies have the most,
   which fits them shaking at least as much as fb2.
5. **The robot's body answers each degree of command 2-2.5× more than the simulator's at 4.5 Hz:** roll rate per
   degree at the roll joints 10-13 on the robot against 4-5 in the fitted simulator, 3.6 for the legs-only robot;
   pitch rate per degree at the knees 6-9 against 3-4.5. Policy gain × body gain round the loop: about 2 on the
   robot, 0.8 in the simulator, 0.4 for b7 on the legs. Above 1 the loop rings by itself, and that is the shake:
   the frame rings at 5 Hz (ζ 0.15), the gyro sees it, the policy answers it at the knees and hip rolls, a fraction of
   a ring late, and the servos ring it again. Every full-body policy so far learned that gain in a simulator where
   the body barely rings, so nothing taught them not to.
6. The 1-in-4 clean steps (§18) are moments when the loop happened to be quiet; the falls come when it ran into a
   step.

### 20.1 Next

1. **Robot-side, no training:** `policy_stand.py run ... --gyro-lpf 2.5` (new, default off) low-passes the gyro the
   policy sees, which cuts the loop gain at 4.5 Hz to ~0.5; `--act-lpf 0.6` on top cuts it further. In the
   simulator, `h4_t05` keeps walking with a 2-3 Hz gyro low-pass (0/6 falls, 38-40 cm) or `act_lpf` 0.4 (0/6, 44 cm);
   `h2_old` gets shakier (2-3/6 falls), so try it on `h4_t05` first. The simulator can't show whether it stops the
   real shake, since it doesn't ring.
2. **Make the simulator ring**, then retrain: a give at the hip pitch, knee and hip roll soft and lightly damped
   enough to ring at 4.5-5 Hz with ζ ~0.15 when tapped (E7), checked by running h2 / fb2 until they shake as on the
   robot. Train from fb2 with the gyro low-pass in the observation (as on the robot), the gyro gain randomised, and
   no hard jump at engage (blend the first action in over ~0.3 s).

### 20.2 The filters on the robot (11:03-11:06)

`h4_t05` four more times, two with `--gyro-lpf 2.5` (11:03, 11:05) and two with `--act-lpf 0.6` on top (11:04, 11:06),
going by the shake numbers. Each walked 3-4 steps before falling.

| trials | lasted (s) | 3.5-8 Hz gyro, roll / pitch (°/s) | commands at 4-8 Hz | fell |
|---|---|---|---|---|
| no filter (10:10) | 12.1, 7.8 | 18-23 / 25-29 | 7% | forward, forward |
| gyro low-pass 2.5 Hz | 26.7, 23.5 | 15-17 / 10-15 | 1.3-2.4% | left, left |
| + action low-pass 0.6 | 30.3, 28.2 | 9-10 / 7 | 0.4-0.5% | left, left |

The loop is broken: the shake is down 3-4× and the walks last 2-3× longer. What is left is a slow forward-back rock at
0.8-1.2 Hz, about once per step, between 13° forward and level (std 3-4°), and every trial now falls to the left.
The knees swing as far as before (23° p5-p95). Both filters add lag (~60 ms at 1 Hz for the gyro filter) the policy
never trained with, which fits the user's "stepping a bit compromised".

## 21. Round 7: a simulator that rings, and policies that stop feeding the ring (2026-11-13 afternoon, simulation)

§20 found the shake to be a loop: the frame rings at 4-5 Hz, the gyro sees it, the policy answers at the knees and hip
pitches, and the robot's body answers each degree at 4-5 Hz 8-10× harder than the simulator's (measuring.md E8). This
round puts that response into the simulator and trains in it. Everything here is simulation; nothing is tested on the
robot yet.

### 21.1 The simulator

- **The frame mode** (`ring` in `EnvConfig` / `EvalConfig`): per axis a damped oscillator driven by the servo targets
  along the pattern that excites it (pitch: both hip pitches; roll: the sideways shift), high-passed, delayed 20 ms,
  its acceleration × 0.02 kg·m² applied to the pelvis. Fitted to the E8 wiggles: pitch 4.2 Hz, ζ 0.2, gain 1.0; roll
  5.3 Hz, ζ 0.25, gain 0.7; with 60 ms of command dead time (the robot's hip pitch follows ~80 ms late) it matches the
  pitch response in size and phase. Randomized per walk in training (frequency ±15%, ζ ×0.6-1.6, gain ×0.6-1.5).
  Tried first and dropped: a softer hip-pitch give (no resonance; the body sags back and falls), the torso on spring
  hinges (`hw_ank3_b150_m2uf_10dof`: the pelvis IMU barely sees it), the give outside the servo's damping.
- **The standing lean as measured:** knee offset 0 to -0.05 rad (the robot stands +5° back at the stance pose; round
  6's -0.09 put the simulated stance past its backward tip point, so the policy had to snap it forward).
- **Two screens:** `eval6.py --sim r7` (the fit) and `--sim r7hard` (stronger, less damped ring, 3 steps of delay,
  deeper lean). On `r7hard` the round-6 policies do what they did on the robot: h4 / h2 / fb2 shake 18-24°/s with
  5-7.5% of their commands at 4-8 Hz and fall in 5 of 6 walks within 7-9 s.
- New options: `gyro_lpf` (gyro low-pass in the observation), `obs_hist` (already there) works with the robot runner.

### 21.2 Results (9 × 30 s at 0.015 m/s; shake = 4-8 Hz gyro, °/s; lifts = foot lifts > 5 mm in 30 s)

| run | from | what's new | r7 falls | forward | shake | commands at 4-8 Hz | r7hard falls | lifts |
|---|---|---|---|---|---|---|---|---|
| `h4_t05` (round 6) | | | 0/6 | 31 cm | 10.1 | 2.0% | 5/6 | 17-18 |
| `h4_t05` + `--act-lpf 0.6` at run time | | as on the robot | 3/9 | 28 cm | 9.4 | 0.9% | 7/9 | |
| `r1_raw` | b7_seed2 | ringing sim | 5/9 | 17 cm | 12.4 | 2.3% | 7/9 | |
| `r1b_raw` | r1 + 40M | | 4/9 | 4 cm | 13.1 | 5.0% | 5/9 | |
| `r3_hist` | c0_hist (b7 + 5-frame history) | ringing sim, history | 4/9 | 13 cm | 12.4 | 4.6% | 6/9 | |
| `r5_histlpf` | c0_hist | history + gyro low-pass 2.5 Hz | 3/9 | 3 cm | 12.1 | 3.6% | 7/9 | |
| `r7_h4` | h4_t05 | ringing sim | 2/9 | 10 cm | 11.5 | 2.5% | 6/9 | |
| `r8_alpf_h4` | h4_t05 | action low-pass 0.6 in training | **0/9** | 3 cm | 7.6 | 0.9% | 3/9 | 3-5 |
| `r9_alpf_hist` | r3_hist | action low-pass + history | 2/9 | 7 cm | 8.9 | 1.9% | 5/9 | |
| **`r10_lcp_h4`** | h4_t05 | Lipschitz penalty 0.0005 | 2/9 | **20 cm** | 9.9 | 1.7% | 5/9 | 19 |
| `r11_alpf_fwd` | r8 | 2× forward pay | **0/9** | 6 cm | **6.4** | 1.1% | **2/9** | 2-9 |
| `r12_alpf5_h4` | h4_t05 | action low-pass 0.5 | **0/9** | 1 cm | 7.7 | 1.2% | **2/9** | 4-6 |
| **`r13_alpf_b7`** | b7_seed2 | action low-pass 0.6 | **0/9** | **17 cm** | 6.9 | **0.7%** | 4/9 | 8-16 |

### 21.3 Learnings

1. **The ring makes the simulator hard in the way the robot is.** The pitch ring plus 60 ms of delay is what tips the
   round-6 policies over in `r7hard` (no ring: 2/6 falls; pitch ring alone: 5/6, all backward).
2. **Reinforcement learning alone doesn't learn to stop feeding the ring.** Fine-tuning in the ringing simulator for
   30-70M steps (`r1`, `r1b`, `r3`, `r5`, `r7`) left the chatter at 2.3-5% and the falls at 2-5 of 9. Neither the
   history (`r3`) nor the gyro low-pass (`r5`) changed that.
3. **What works is lowering the policy's gain round the loop by construction:** the action low-pass in training
   (`r8`, `r11`-`r13`: no falls on `r7`, shake 6.4-7.7, commands at 4-8 Hz ≤ 1.2%) or the Lipschitz penalty (`r10`).
   The filter also makes the ringing simulator learnable from the legs policy (`r13` reached episodes of 708 control
   steps; `r1`, same start without it, 337).
4. **But the filtered policies from h4 learned to stand:** `r8`, `r11`, `r12` lift a foot 2-9 times in 30 s and move
   1-6 cm; twice the forward pay (`r11`) didn't bring the steps back. From the legs policy (`r13`) the steps stay:
   8-16 lifts, 17 cm.
5. The simulator's shake is still milder than the robot's at the `r7` fit (h4: 10 against 20-30°/s on the robot);
   `r7hard` is closer. The robot will tell which is right.

### 21.4 What to try on the robot

Full body, `--vx 0.015`, cables slack. The runner applies each run's trained filter (`act_lpf`) by itself.
1. **`r13_alpf_b7`**: the steadiest that still steps (b7's gait, action low-pass 0.6).
2. **`r10_lcp_h4`**: h4's steps with less chatter (no filter).
3. `r11_alpf_fwd`: the calmest, mostly stands: tells whether the twitch is gone.
New runner option for each: `--ready-act` blends into the pose the policy holds standing (gait start pose + its action)
so it doesn't snap the body forward at engage; worth one trial with and one without (untested on the hardware).
Watch: the 4-5 Hz shake (does it start at engage?), whether it steps, and the fall direction.

## 22. Round 7 on the robot: the ring is tamed, the steps tip forward (2026-11-13 evening)

Full body, `--vx 0.015`, cables slack, 19:01-19:14 (logs `stand_*_20261113_19*.csv`). The flags of each trial are
confirmed from the logs where they leave a trace (the action low-pass: replaying the policy reproduces the logged
actions with k = 0.60 exactly) and otherwise from the user.

| policy | trials (s) | steps (lifts > 5 mm) | walking pitch (+ = back) | 4-8 Hz gyro roll / pitch (°/s) | fell |
|---|---|---|---|---|---|
| `r13_alpf_b7` | 6.0, 9.0, 9.2, 8.6 | 5-10 | -7.8 to -8.8 | 7-8 / 8-9 | forward, every trial |
| `r10_lcp_h4` | 14.9, 5.0, 9.7, 6.6, 38.3 | 6-71 | -9.4 to -12.3 | 9-10 / 4-6 | forward, every trial |
| `r11_alpf_fwd` | 30.8 | 47 | -7.4 | 9 / 7 | forward |
| `h2_old` (filtered) | 13.1 | 15 | -6.3 | 13 / 12 | forward |
| **`h4_t05 --act-lpf 0.6`** | **188.1, 74.3, 75.6** | **299, 115, 109** | **-4.4 to -4.7** | 15-16 / 12 | ended upright; fell left; ended at 7° roll |
| `h4_t05 --gyro-scale 0.75` | 3.1, 3.8 | 1-3 | -14 to -16 | (too short) | forward, within 2 s |

For comparison, the morning's unfiltered h4: 18-23 / 25-29°/s of shake, 8-12 s.

The story:

1. **The action filter in training did what it was for.** The round-7 policies let the 5 Hz ring die down instead of
   answering it: 4-8 Hz gyro 7-10 / 4-9°/s, half of h4 with the run-time filter and a third of unfiltered h4. The user
   saw the same: no fighting the vibration.
2. **But every one of them leans forward and catches its toes.** Every policy, old or new, snaps the body from its
   standing +5° back to about -3° in the first second after engaging (the snap of §20). h4 then settles at -5°; `r10`
   and `r13` keep going to -9° within 2-4 s and stay there, the boards tip onto their toes, the toes kick the rug,
   and all 10 trials of the round-7 policies ended face down.
3. **The simulator doesn't see the lean.** On the robot their hip pitch and knee angles match the simulator's to a
   degree (hip ~19-21°, knee ~29-31°, like h4's), and in the simulator they walk at -4.5° like h4. The extra 3-5°
   forward on the robot is the boards rolling onto their toes, not a different leg posture: a dynamic effect of their
   gait that the round-7 simulator (frame ring, 60 ms dead time, standing lean 0 to -0.05 rad at the knees instead of
   round 6's 0 to -0.09) doesn't reproduce. Which of the three is to blame is untested.
4. **The run-time filter on a policy trained without it is the best combination so far.** `h4_t05 --act-lpf 0.6`
   walked 188 s (299 steps), 74 s and 76 s: the first full-body walks of minutes. h4 keeps the steps and the 4.5°
   lean it learned in round 6; the filter takes enough of the 5 Hz loop gain away. Its remaining fault is the
   occasional bout of twitching against the ring (gyro 15-16°/s).
5. **Turning the gyro down instead (`--gyro-scale 0.75`) failed twice in 3-4 s,** both pitching 14-16° forward right
   after the engage snap (earlier the same flag gave 16 and 42 s).

### 22.1 Next

1. Keep `h4_t05 --act-lpf 0.6` as the base. Fine-tune it with the filter in training on the round-6 simulator (whose
   lean it walks well with) plus the ring, then on the round-7 stance, one change at a time, to find what tips the
   round-7 policies forward.
2. Penalize forward pitch past ~6° in training (the toe catch), and soften the engage snap (`--ready-act` on the
   robot, untested).
3. A twitch measure: 4-8 Hz command share while standing between steps, to compare h4+filter and its fine-tunes.

## 23. Round 8: what tips the round-7 policies forward, and h4 with the filter trained in (2026-11-13 night)

Simulation only (1.5 h of GPU). Scripts in `scripts/robot8_analysis`, notes and screens in `results/robot8`.

### 23.1 The forward lean

- **Standing, the robot and the simulator agree; walking, only the robot rolls onto its toes.** Before engage every
  policy stands at +4-7.5° with knee - hip pitch 17.2° (the simulator gives that at a knee offset of about -0.04 rad).
  Walking, the foot proxy pitch - (knee - hip) stays at -12 to -15 in the simulator for every policy on every screen;
  on the robot it is -13 to -17 for h4, -16 to -21 for r13 and -18 to -22 for r10. r10 also walks with its knees
  unloaded and its hip pitches holding the torso (servo load +54 / -24 against h4's +6 / -2.5).
- **Only a forward bias reproduces it.** Knee offsets 0 to -0.09, a soft rug, weak servos, a harder ring, no ring,
  low friction, 20% more mass: none tips h4 / r10 / r13 past -5 to -7.6°. The pelvis centre of mass 1-2 cm further
  forward (new `com_dx` in `EvalConfig`), or a 0.3 N forward push, does: -7.7 to -9.2°, knee - hip down to 8-9, the
  feet onto their toes, forward falls at 3 cm; and the 0.5 s engage snap grows to the robot's -6 to -9°.
- **The real mass probably sits ~1-2 cm further forward than the model's.** With zero action at the robot's standing
  pitch the model tips backward in 5 of 6 draws, 3 of 6 with the mass 1 cm forward, none at 2 cm; the robot stands
  there for seconds. No training run had this bias.
- **Why h4 copes and the round-7 policies don't:** r10's Lipschitz penalty cut its posture answer to a lean to a
  quarter of h4's (knee - hip -0.14 against -0.57° per degree of tilt), so it sags furthest. r13 / r11 answer as
  hard as h4 and still lean: that part is the gait they learned in the round-7 simulator, not pinned down.

### 23.2 Training the filter into h4 (from h4_t05, `act_lpf` 0.6, pelvis mass +[-0.5, 2.5] cm: new `dr_com_x`)

Screens: 6 × 30 s each on the ring fit (`r7`), `r7` with the mass 1.5 cm forward, the round-6 fit with it, and `r7hard`.
Gains measured on h4's 188 s robot walk (`gains.py`).

| run | what | r7 falls / fwd / shake | r7 + 1.5 cm | fitted + 1.5 cm | r7hard falls / fwd / shake | lifts / 30 s (r7, +1.5 cm) | gyro gain hip / knee |
|---|---|---|---|---|---|---|---|
| `h4_t05 --act-lpf 0.6` | the robot's best | 1/6, 37 cm, 8.3 | 1/6, 25 cm | 0/6, 20 cm | 5/6, -5 cm, 16.5 | 25, 17 | 0.047 / 0.096 |
| `h4_t05 --act-lpf 0.7` | stronger run-time filter | 1/6, 33 cm, 7.9 | 2/6, 25 cm | 0/6, 24 cm | 5/6, -11 cm, 13.9 | | same |
| `s1_alpf_com` | filter trained in, round-6 sim, mass forward | 1/6, 30 cm, 10.2 | 3/6, 31 cm | 0/6, 37 cm | 6/6 in 2.8 s | 23, 22 | 0.090 / 0.161 |
| `s2_alpf_com_ring` (11M) | s1 + the ring and 60 ms | 0/6, 17 cm, 7.4 | 1/6, 3 cm | 0/6, 5 cm | 2/6, 0 cm, 14.0 | 15, 8 | 0.060 / 0.101 |
| `s3_swing` 12M | s2 + swing pay 15 (6), contact pay 2 (3) | 0/6, 30 cm, 8.0 | 2/6, 15 cm | 3/6, 28 cm | 2/6, 12 cm, 12.4 | 19, 16 | 0.071 / 0.116 |
| **`s3_swing` 24M** | | **0/6, 24 cm, 7.4** | **1/6, 8 cm** | **1/6, 17 cm** | **2/6, 7 cm, 11.5** | 17.5, 14 | 0.074 / 0.132 |

Learnings:
1. **Trained in without the ring, the filter is undone:** s1 doubled its gyro and tilt gains to make up for the lag
   and is worse than h4 + the run-time filter on every ringing screen.
2. **With the ring, the gains stay near h4's but the steps fade** (s2, as r8 in round 7): on the step-settle-step
   schedule both feet are down most of the time, so standing collects most of the contact pay (r8: contact pay
   1572 -> 1961, swing 310 -> 115, stride 190 -> 35).
3. **Paying more for the swing brings the steps back** (s3: 15 -> 19 lifts, 17 -> 30 cm at 12M) and keeps the ring
   tolerance: s3 is the first that walks forward on `r7hard`, with the least shake there. Its gains creep up again
   (1.5× h4's at 24M), and with the mass forward it still walks shorter than h4.
4. The screens are noisy at 6 walks (h4 + filter falls 1/6 or 3/6 on `r7` depending on the seeds); the differences
   on `r7hard` and in the lifts are the solid ones.

### 23.3 For the robot

1. `h4_t05 --act-lpf 0.7` (no retraining; a little calmer than 0.6 in simulation).
2. `s3_swing` (24M; the runner applies its trained filter 0.6): the steadiest that still steps.
   `--params params_0012M.pkl` for the earlier one that steps more.
3. `s2_alpf_com_ring` (its newest checkpoint is 11M): steady, few steps; tells whether the twitch is gone.
Watch: the twitch while standing, the forward lean and toe catches, step count.

## 24. Round 8 on the robot: h4 + filter 0.7, s3 twitches, s2 steps in place (2026-11-13 22:16-22:30)

Full body, cables slack. Flags confirmed by replaying each policy against its log (exact: residual 0): the h4 trials
used `--act-lpf 0.7`; the first two `--vx 0.015`, the last three `--vx 0.02`; s2 / s3 their trained filter 0.6.
Measured with `scripts/robot8_analysis/twitch.py` and `falltrace.py`: 4-8 Hz gyro (deg/s) with both boards down and
while one swings, 4-8 Hz commands (deg), share of time in a bout (4-8 Hz gyro > 25 deg/s over 0.5 s).

| trial | time, steps | 4-8 Hz gyro roll / pitch, both down / swinging | 4-8 Hz cmd hip roll / hip pitch / knee | in a bout | end |
|---|---|---|---|---|---|
| h4 0.7, vx 0.015 | 147 s, 251 | 14.2 / 11.6, 14.3 / 9.4 | 0.31 / 0.48 / 0.76 | 11% | standing |
| h4 0.7, vx 0.015 | 138 s, 227 | 13.5 / 11.2, 13.4 / 9.8 | 0.29 / 0.47 / 0.74 | 9% | standing |
| h4 0.7, vx 0.02 | 84 s, 147 | 19.7 / 13.6, 16.8 / 11.4 | 0.34 / 0.58 / 0.99 | 28% | standing |
| h4 0.7, vx 0.02 | 67 s, 117 | 16.9 / 12.2, 15.3 / 9.8 | 0.33 / 0.53 / 0.90 | 16% | standing |
| h4 0.7, vx 0.02 | 32 s, 63 | 18.8 / 12.9, 18.4 / 10.3 | 0.36 / 0.58 / 1.01 | 22% | fell right, in a bout |
| h4 0.6, vx 0.015 (§22, 3 trials) | 188 / 74 / 76 s | 13.3-14.8 / 11.4-12.2, 12.2-13.2 / 10.5-11.9 | 0.39-0.41 / 0.58-0.62 / 0.95-0.97 | 9-15% | standing / fell left / 7° roll |
| `s3_swing` | 38 s, 81 | 18.5 / 18.0, 20.4 / 20.8 | 0.52 / 0.76 / 1.30 | 51%, from 2.5 s | fell left, in a bout |
| `s2_alpf_com_ring` | 46 s, 59 (in place) | 13.9 / 14.9, 11.9 / 13.7 | 0.36 / 0.58 / 0.83 | 9%, from 20 s | standing, pitch -7 |

The story:

1. **The stronger filter calms the commands but hardly the body.** At 0.7 h4's 4-8 Hz commands are 20-25% smaller
   than at 0.6, yet the body's 4-8 Hz shake is the same (13.5-14 / 11-12°/s) and it still spends ~10% of the time in
   bouts. A floor of ~12-14°/s doesn't follow the commands: the steps and landings ring the frame by themselves.
   The bouts on top are the loop taking over. At 0.015 both trials walked over 2 minutes (227-251 steps) and ended
   standing: as good as 0.6, not better.
2. **Faster is shakier.** At `--vx 0.02` the shake rises to 17-20°/s roll and bouts to 16-28% of the time; the one fall
   of the five came from a bout: calm (9-10°/s) 3 s before, then within one step the shake grew to 50°/s in a second,
   the roll swung +8° to -7° and then over to the right.
3. **The twitch is as strong standing as stepping** (h4: 14.2 against 14.3°/s roll with both boards down vs one
   swinging; pitch 11.6 against 9.4): the policy answers the ring when there is nothing to do.
4. **s3 twitches worst, as its gains said and the simulator didn't.** Its 4-8 Hz commands are ~1.6× h4's, the body
   shakes 18-21°/s on both axes and is in a bout half the time from 2.5 s on; it fell left in the middle of a
   30-37°/s bout. The gyro gain measured on h4's robot walk (§23: s3 0.074 / 0.132 against h4 0.047 / 0.096)
   predicted this; the simulator screens (`r7hard` 2/6 against h4's 5/6) ranked it the wrong way round. The
   simulated ring is still milder than the robot's (sim shake 7-12°/s, robot 13-21).
5. **s2 has h4's gains and h4's shake, but no steps:** 59 steps in place in 46 s, bouts 9% of the time and only from
   20 s, roll within 5.6°. Leaning a little further forward than h4 (-5°, 10th percentile -9°).

### 24.1 Learnings and next

- **The gyro gain on a robot log is the better twitch predictor** than the ringing simulator: rank candidates by it
  before the robot. A policy whose gyro gain went up in fine-tuning will twitch more.
- **Fine-tuning tends to raise the gyro gain** (r8, s1, s3) unless something holds it. Next: a gain penalty on the gyro
  inputs only (the Lipschitz penalty of r10 on every sensor input also cut the posture answer to a lean, which let
  r10 sag forward), with the ring and the larger swing pay of s3, from h4.
- The landing-driven floor of the shake won't go with any filter on the policy; softer landings (swing-down speed,
  `clear`, landing impact) are the other lever.
- On the robot, stay at `--vx 0.015` with h4 + `--act-lpf 0.6` or `0.7`.

## 25. Round 9: a faster continuous gait from h4, and a breadth-first search on the shake loop (2026-11-14 01:50-09:50)

Goal: walk faster than h4 while twitching less; small balancing moves allowed but economical; break the loop where a
small vibration makes the policy react and the reaction grows the shake. 8 h of GPU, one policy per idea, all
screened in simulation; the robot decides. Full log: [results/robot9/NOTES.md](../results/robot9/NOTES.md).

### 25.1 The gait and the base

`c_T10L` (`shift_gait step --T-step 1.0 --settle0 0 --shift-t 0.40 --swing-t 0.45 --release-t 0.15`, otherwise as
`c_T12`): 1.0 s steps instead of 1.2, no pause before or after the swing, the weight shift flowing straight into the
next one (both feet down 55% of the time, was 62%). Trained at 0.02-0.04 m/s (2-4 cm steps); **robot command
`--vx 0.027`** = 2.7 cm steps, 1.5x h4's 1.8 cm reference step at 0.015.

Why h4 was slow: it never followed the speed command (simulation: ~1.2 cm/s at every command 0.015-0.04, fewer
metres at 0.04 than at 0.015). Its stride pay was a fixed 5 cm target and standing in the reference pose earned more
than walking. New: `stride_cmd` (stride paid up to the commanded step length), ref_q 5 with a tighter kernel
(`ref_sigma_q` 0.05), `ref_tip` 2, and after the first two branches drifted back to standing (d1: progress pay
542 -> 319; d2 with the ring: 580 -> 151), progress 6, ref_swing 12, ref_contact 2 for every later branch.

Base `g1_t10l`: h4 -> c_T10L on h4's own simulator (no ring), 40M. Screen at vx 0.027, run-time filter 0.6:
46.5 cm / 30 s on h4's simulator with the pelvis mass 1.5 cm forward (h4: 12 cm at this command, 20 at its 0.015).

### 25.2 What others did (web, 2026-11-14)

No project describes a frame resonance fed by the policy through the gyro. Related: action smoothness losses (CAPS,
arXiv 2012.06644; Grad-CAPS 2407.04315; LCP 2410.11825; spectral normalization 2504.08246; the benchmark 2410.16632);
a quadrotor whose software gyro filter *caused* oscillation through its lag (1903.04628); delay randomization smoothing
stair policies (2402.06143); a lower policy rate letting joints settle (2604.02523); hobby-servo bipeds (Microduck /
Open Duck) modelling backlash and filtering actions at 4 Hz; and "Too Much of a Good Thing" (2606.02636): heavy sim2real
fidelity can make policies conservative and hurt learning.

### 25.3 The branches (from g1 at 40M, 28M each)

Screen (`scripts/robot9_analysis/screen9.sh`): 6 x 30 s at vx 0.027; "h4 sim" = h4's simulator + pelvis mass 1.5 cm
forward, "ring" = round-7 ring + 1.5 cm; shake = 4-8 Hz gyro, deg/s; gain = pitch-rate gain hip / knee on h4's 188 s
robot log (round 8: predicted the robot's twitch ranking). Run-time filter 0.6 unless trained with one.

| run | idea | h4 sim: falls, cm, shake | ring: falls, cm, shake | r7hard | gain |
|---|---|---|---|---|---|
| h4_t05 (vx 0.015) | reference | 0/6, 20, 7.0 | 1/6, 25, 9.1 | 5/6 | 0.047 / 0.096 |
| g1_t10l | base | 0/6, 46.5, 7.6 | 3/6, 32, 10.5 | 6/6 | 0.043 / 0.102 |
| d1_ignore | no ring, old pay | 0/6, 27, 6.8 | 1/6, 25, 8.5 | 4/6 | 0.055 / 0.100 |
| **d1b_ignore** | no ring: ignore the problem | 0/6, **57**, 7.0 | 2/6, 42, 9.6 | 6/6 | 0.053 / 0.114 |
| d2_ring05 | half ring, old pay | 0/6, 6.5, 6.7 | 0/6, 6, 6.4 | 4/6 | 0.052 / 0.103 |
| d2b_ring05 | half ring | 0/6, 30, 8.1 | 3/6, 31, 9.9 | 4/6 | 0.070 / 0.093 |
| d3_ring10 | round-7 ring | 4/6, 17, **20.5** | 3/6, 21, 8.8 | 3/6 | 0.050 / 0.086 |
| d4_ring15 | 1.5x ring | 3/6, 12, **32.5** | 5/6, 21, 8.3 | 4/6 | 0.032 / 0.058 |
| d5_hf | filter trained in + raw-output penalty > 2 Hz | 0/6, 54, 8.7 | 2/6, 33, 10.7 | 4/6 | 0.093 / 0.119 |
| d6_glcp | gyro-only gradient penalty 0.05 | 0/6, 21, 6.0 | 4/6, 24, 7.9 | 4/6 | **0.011 / 0.016** |
| d6b_glcp01 | gyro-only gradient penalty 0.01 | 1/6, 20, 7.7 | 1/6, 15, 8.1 | 4/6 | 0.028 / 0.049 |
| d7_vib | fake 3.5-7 Hz gyro bursts | 0/6, 28, 8.3 | 1/6, 8, 8.1 | 4/6 | 0.059 / 0.100 |
| d8_glpf | 2.5 Hz gyro low-pass trained in | 0/6, 23, 6.9 | 3/6, 23, 8.2 | 5/6 | 0.059 / 0.082 (x0.45 at 5 Hz) |
| d9_blind | no gyro | 4/6, 18, 8.9 | 5/6, 20, 11.7 | 5/6 | (gyro unused) |
| d9c_blind | d9 + 28M | 4/6, 16, 9.0 | 3/6, 7, 11.3 | 6/6 | (gyro unused) |
| d10_land | soft touchdowns | 0/6, 22, 7.6 | 1/6, 22, 8.3 | 4/6 | 0.064 / 0.108 |
| d11_combo | d7 + d5 + d10 | 0/6, 38.5, 7.6 | 1/6, 36, 8.2 | 4/6 | 0.094 / 0.111 |

d2-d11 except d3 / d4 train on the half ring.

### 25.4 Learnings

1. **The gait and the pay fix the speed, not the shake work.** d1b walks 1.9 cm/s on the forward-mass simulator,
   4.7x h4 at the same command, with h4's shake and roughly h4's gains. Every branch on a ring is slower (15-38 cm).
2. **A stronger training ring makes a policy that shakes a body that rings differently.** On the no-ring simulator:
   d1b 7.0, d2b 8.1, d3 20.5, d4 32.5 deg/s, with *lower* static gyro gains. That is the round-7/8 robot failure
   reproduced in simulation: the policy learns a counter-move timed to the training ring, and on another ring the same
   move feeds it. The static gyro gain misses this because the loop's phase depends on the body.
3. **Training the filter in doubles the hip's gyro gain** (d5, d11 0.093-0.094; s1 likewise): the policy pushes
   harder behind the filter. The raw-output penalty didn't stop it, nor did the fake vibration (the policy has no
   memory, so it can't tell a fake 5 Hz wobble from a real tilt rate).
4. **Only the gyro-only gradient penalty cuts the gain directly** (d6 -80%, d6b -50%) without touching the posture
   answer much (d6b tilt -0.48 vs h4 -0.57). Both are slow, as all half-ring branches are.
5. Soft touchdowns (d10) didn't lower the simulated shake floor; the gyro low-pass (d8) cuts the 5 Hz gain to ~40%
   at the cost of lag; going blind (d9) needs much longer training.

### 25.5 For the robot (all at `--vx 0.027`; try 0.02 first if it hesitates)

`python policy_stand.py run --run <run> --enable --vx 0.027 [--act-lpf 0.6]`. The runner takes the gait clock,
settle gate, gyro low-pass, trained-in action filter and gyro-blindness from each run's config; give `--act-lpf 0.6`
(or 0.7) to the runs trained without a filter.

1. `d1b_ignore --act-lpf 0.6`: ignore the problem; the fastest; the main test of the new gait.
2. `g1_t10l --act-lpf 0.6`: the base.
3. `d6b_glcp01 --act-lpf 0.6` and `d6_glcp --act-lpf 0.6`: gyro gain cut 50% / 80%: the twitch test.
4. `d11_combo` (filter from its config): the steadiest ring branch.
5. `d5_hf` (filter from config), `d8_glpf --act-lpf 0.6` (gyro low-pass from config), `d10_land --act-lpf 0.6`,
   `d7_vib --act-lpf 0.6`.
6. The ring sweep: `d2b_ring05`, `d3_ring10`, `d4_ring15` (all `--act-lpf 0.6`); d3 / d4 shook on the no-ring
   simulator, so expect them to shake on the robot.
7. `d9c_blind --act-lpf 0.6` (no gyro, from its config).

## 26. Round 9 on the robot: the new gait tips sideways, the gyro penalty transfers, and why h4 works (2026-11-14 11:04-11:32)

Full body, cables slack, `--vx 0.027`, `--act-lpf 0.6` on the runs trained without a filter. 28 trials, logs
`stand_*_20261114_*.csv`; measures from `scripts/robot8_analysis/twitch.py`; per-trial table in
`results/robot9/NOTES.md`. h4_t05 + filter repeated as the reference: 108 s and 47 s.

| policy | trained (all from g1 except h4; g1 from h4) | 4-8 Hz gyro r / p, °/s | longest | the user's note |
|---|---|---|---|---|
| **h4_t05** | round 6, c_T12, gate up to 1 s | 13-17 / 10-14 | **108 s** | still the best |
| g1_t10l | c_T10L, tighter pose match, progress ×2 | 16-20 / 9-13 | 12 s | twitch, tips sideways at once |
| d1b_ignore | g1 + progress 6, swing pay 12, no ring | 20-31 / 13-23 | 18 s | bigger steps, toes catch, heavy twitch |
| d2b_ring05 | half ring, 2-3 steps of delay | 14-23 / 21-25 | 12 s | sideways |
| d3_ring10 / d4_ring15 | round-7 ring / 1.5× | 11-13 / 10-13; 4 / 5 | 4 s / 3 s | sideways and forward rocking |
| d5_hf | half ring + raw-action penalty + filter trained in | 17 / **34** | 7 s | massive twitch |
| d6_glcp | half ring + gyro-only LCP 0.05 | 7-8 / 4-7 | 3 s | forward / back rocking, falls forward |
| **d6b_glcp01** | half ring + gyro-only LCP 0.01 | **9-11 / 6** | 3 s | **no twitch**, tips sideways |
| d7_vib | half ring + fake 3.5-7 Hz gyro vibration | 16 / 13.5 | 34 s | a few steps, vibrating |
| d8_glpf | half ring + 2.5 Hz gyro low-pass | 13 / 9 | 11 s | little twitch, sideways |
| d9c_blind | half ring, no gyro | 9 / 4.5 | 3 s | instant side twist |
| d10_land | half ring + landing-speed penalty | 18 / 23 | 14 s | stuck in place fighting the shake |
| d11_combo | d5 + d7 + d10 | 27-31 / **33-38** | 32 s | massive twitch |

### 26.1 What the logs show

1. **The sideways falls are the gait, not the twitch.** Nine trials (g1, d4, d6 ×2, d6b ×3, d7, d9c) end within 3.1 s
   at the third lift-off, ~2.2 s after engage. The first two steps match h4's. After the second landing h4 lets the
   roll die out (+0.6° at 2.5 s) and lifts ~0.5 s after landing; the c_T10L policies are still rolling +3 to +7°
   and lift ~0.3 s after landing, so the landing roll and the next weight shift add up and carry the body past the
   stance foot (+19° within 0.5 s). The quiet policies fall the same way as the twitchy ones. c_T10L cut the release
   (0.25 → 0.15 s), the settle before the first step (0.05 → 0) and the settle gate's longest hold (1.0 → 0.2 s);
   the simulator doesn't charge for any of that.
2. **The twitch ranks as the simulator said:** gyro LCP and blind lowest, then the gyro low-pass, then h4 / g1 / d7,
   then the heavier pay (d1b), then the filter trained in (d5, d11) worst. The gyro-only LCP is the first training
   change that cut the twitch on the robot: d6b's 4-8 Hz commands are about half h4's.
3. **The gyro gain has a floor.** d6 (LCP 0.05) and d9c (blind) lost the fast pitch damping and fell forward or
   twisted at once; d6b (0.01) kept it.
4. **More pay, more twitch.** d1b differs from g1 only in reward weights and shakes ~30% more in roll.
5. **The ring in training didn't help on the robot at any strength** (d2b-d4).
6. **The landing-speed penalty taught hesitation** (d10), and hesitating while shaking is the worst of both.

### 26.2 Why h4 works

| what h4 has | why it matters | the evidence |
|---|---|---|
| **No ring in training** (round 6: measured masses, series give, no ring) | Its gyro answer is generic, not timed to a simulated ring; the run-time filter can then cut the loop where it rings | Every ring-trained policy (rounds 7-9) shook more or stepped less on the robot; on the no-ring simulator, shake grows with the training ring (§25.4) |
| **Light reward pressure**: tilt penalty -0.05, progress 1.5, loose pose match (σ 0.1) | Nothing pushes it to answer every disturbance; its round-6 commands were the smoothest (2.3% at 4-8 Hz against 4-11%) | d1b vs g1: heavier pay, ~30% more shake; round 6's stronger tilt penalties chattered more (§19.3) |
| **No filter in training, filter at run time** | The training still punishes jerky actions (the rate and second-difference penalties see the raw action), and the run-time filter removes only the 5 Hz part | Every policy with the filter trained in (s1, s3, d5, d11) doubled the hip's gyro gain and twitched worst |
| **A gait that waits for calm**: c_T12, 0.25 s release, gate holds up to 1.0 s while the body turns faster than 0.3 rad/s | The landing roll dies out before the next weight shift | §26.1 item 1: the c_T10L policies lift while still rolling and fall over the stance foot |
| **A strong slow posture answer** (knee - hip -0.57° per degree of lean) | It catches the slow lean while the filter cuts the fast part | r10 (LCP on all inputs) cut it to a quarter and fell forward; d6 and d9c fell at once |

### 26.3 What h4's training could do better

| weakness | cause in training | fix | evidence for the fix |
|---|---|---|---|
| ~1.2 cm/s at any speed command | a fixed 5 cm step target; standing in the reference pose pays about as much as walking | `stride_cmd` (step target from the command), at h4's own pay level | round 9: the per-command target gave the speed (d1b 4.7× h4 in simulation) |
| Twitch bouts 9-21% of the time; they end most trials | gyro gain 0.047 / 0.096 (hip / knee, deg per °/s) | gyro-only LCP 0.01 (`--lcp 0.01 --lcp-idx 3:6`) | d6b: half h4's 4-8 Hz commands on the robot, pitch damping kept |
| Leans 4.5° forward | the model's centre of mass 1-2 cm behind the robot's | `dr_com_x` [-0.005, 0.025] | §23: matches the robot's lean in simulation |
| Toes catch, more so with longer steps | 15 mm swing height in c_T12 | a reference that lifts higher (penalising low clearance alone doesn't teach lifting) | d1b's toe catches; §2: the reference sets the step |
| A jolt at engage (6° at a hip pitch on the first tick) | training starts from the full action | blend the first actions in | §20: the shake starts on the first tick |
| Can't tell the ring from a real rotation | no observation history | 2-3 frames of history (experimental) | d7's fake vibration taught nothing to a memoryless policy |

Keep: c_T12's timing and 1 s gate, no training ring, no filter in training, the light tilt penalty, the loose pose
match, `--act-lpf 0.6` at run time. Unresolved: h4 trained with 0-1 steps of delay, the ring branches with 2-3;
round 9 can't separate the two.

Next (round 10, §27): single changes from h4 (speed, gyro LCP, centre of mass, higher steps), their combination, and
an open-loop-within-a-step policy.

## 27. Round 10: single changes from h4, higher steps, and open loop within a step (2026-11-14 13:13-16:07, simulation)

All from `h4_t05` at 30M on h4's own recipe (§26.2's "keep" list: c_T12 timing, the settle gate holding up to 1.0 s,
no ring, no filter in training, the light tilt penalty, the loose pose kernel). Queue `scripts/robot10_queue.sh`,
settings `scripts/robot10_common.sh`, per-run notes `results/robot10/NOTES.md`.

New:
- **Higher-stepping references** `c_T12H` / `c_T12HH`: c_T12's timing with `--lift-knee`/`--lift-hip` 0.60 / 0.75
  (was 0.45). First-step foot clearance played open loop at 0.02 m/s: 27.5 / 39 / 59 mm. `ref_swing_h` raised to
  18 / 25 mm with them.
- **Open loop within a step** (`obs_hold_u`, in `mjx_env`, `evaluation` and `policy_stand`): the policy's sensor
  inputs (gravity, gyro, q, 0.1 qd) are sampled when the gait clock passes the given cycle fractions and held in
  between; the clock, command and last action stay live, and the settle gate still reads the live gyro.
  `obs_hold_keep_g` keeps gravity live. While the speed command is zero the inputs are live.
- `scripts/robot10_analysis/clear.py`: each swing's peak foot height in simulation.

### 27.1 Results (6 × 30 s at 0.02 m/s, `act_lpf` 0.6 at run time; fitted / r7 with the pelvis mass 1.5 cm forward)

| run | change from h4 | fitted: falls, cm, shake | r7: falls, cm | r7hard falls | gyro gain hip / knee | swing peak (median) |
|---|---|---|---|---|---|---|
| `h4_t05` | | 0/6, 16, 7.3 | 2/6, 25 | 4/6 | 0.047 / 0.096 | 6.7 mm |
| `k1_speed` | step target from the command | 0/6, 14, 6.4 | 3/6, 20 | 4/6 | 0.046 / 0.100 | |
| `k2_glcp` | gyro-only LCP 0.01 | 0/6, 14, **5.2** | 3/6, 19 | 5/6 | **0.025 / 0.051** | 5.9 |
| `k3_com` | centre of mass -0.5..+2.5 cm | **0/6, 42**, 7.5 | **0/6, 33** | 6/6 | 0.051 / 0.111 | 9.8 |
| `k4_high` | c_T12H | 1/6, 28, 8.8 | 4/6, 30 | 6/6 | 0.049 / 0.102 | **15.0** |
| `k5_high2` | c_T12HH | 1/6, **49**, 7.8 | 6/6, 21 | 6/6 | 0.047 / 0.103 | **33.2** |
| **`k6_combo`** | k1 + k2 + k3 + c_T12H | **0/6, 46**, 7.5 | 5/6, 24 | 6/6 | 0.031 / 0.055 | **16.3** |
| `o1_hold4` | sensors held between lift-offs and touchdowns | 2/6, 11, 7.4 | 4/6, 7 | 4/6 (shake **10.7** vs 16.5) | (held) | 4.3 |
| `o3_hold4g` | o1, gravity live | 3/6, 1, 11.7 | 4/6, 15 | 5/6 | (held) | |
| `o2_hold2` | sensors held a whole step | 4/6, 16, 9.3 | 6/6, 13 | 5/6 | (held) | |

### 27.2 Learnings

1. **A higher reference teaches a higher step.** h4 lifts its feet a quarter of its reference's height (6.7 mm
   median, 75% of swings under 10 mm), despite the clearance penalty in every run. Training on a reference that
   lifts 1.4× / 2.1× higher raised the real lift 2.2× / 5× (15 / 33 mm) and lengthened the steps; they also roll
   more and fall more on the ring screens, and were still adapting at 16M.
2. **The step target following the command doesn't buy speed on its own** at h4's reward weights (k1: no faster at
   0.02 or 0.03 m/s). Speed came from the centre of mass range (k3) and the higher steps (k4, k5).
3. **The centre of mass range gave the best ring screen of any policy** (k3: 0/6 on r7 at 33 cm). Partly flattering:
   the screens' 1.5 cm forward mass is inside its training range.
4. **The gyro-only LCP halves h4's gyro gain again** (k2, as d6b did from g1) with the steps kept; it does not
   lower the clearance problem, and combined with the rest (k6) the gain is still 35-43% below h4's.
5. **Open loop within a step lowers the shake but costs the slow balance.** o1 shook least of all on the hard
   ring (10.7 against h4's 16.5°/s, same falls), but its posture answer halved (knee - hip -0.30° per degree of
   tilt against -0.57) and it walks 7° forward and shuffles (4 mm lift). Holding for a whole step (o2) is worse;
   keeping gravity live (o3) didn't bring the posture answer back. All three were still learning (episodes
   330-510 of 1000 in training); they need longer training or a live slow channel (gravity low-passed).

### 27.3 For the robot (`--vx 0.02 --act-lpf 0.6` for all; cables slack)

`python policy_stand.py run --run <run> --enable --vx 0.02 --act-lpf 0.6`

1. `k6_combo`: the strongest simulated walker; lower gyro gain, 16 mm lift.
2. `k3_com`: the best ring screen; h4 with the mass forward.
3. `k2_glcp`: h4 with half the gyro gain (the twitch test).
4. `k4_high`, then `k5_high2`: the higher-step test (toe catches); expect more roll.
5. `o1_hold4`: open loop between steps. The runner prints "sensor inputs held between clock phases"; untested on
   the robot. Then `o3_hold4g`, `o2_hold2`.
6. `k1_speed`.

## 28. Round 10 on the robot: high steps transfer, open loop removes the twitch, and why the steps tip (2026-11-14 16:43-18:20)

All nine round-10 policies on the rug, cables slack, `--vx 0.015-0.02 --act-lpf 0.6-0.7`. Operator notes, the
measured table and the method: `results/robot10/NOTES.md`; scripts `scripts/robot10_analysis/{engage,steps,liftoff,falls}.py`.

| run | on the robot |
|---|---|
| **`k5_high2`** | **closest to the next milestone** (best at `--act-lpf 0.7 --vx 0.015`): consistent large steps (~2x h4's lift), limited shaking, until it tips sideways; every step risky. 12 trials, 4-20 s |
| `k4_high` | the same kind of step, smaller; 7 falls, all sideways, 6 of them lifting the left foot |
| `o1_hold4` | falls on its face at engage; held through that, **the twitch is practically gone**. Sways (strafes) sideways on the loose hips, marches mostly in place, never fell (130 s) |
| `o3_hold4g` | no twitch, walks with h4-like steps, but a forward/back rock builds over ~6 steps and it falls |
| `o2_hold2` | falls on its face at the first step |
| `k3_com` | like h4, less stable; the right leg is passive (lifts half the left's height) and catches the rug |
| `k1_speed` | like h4, slightly less stable |
| `k2_glcp`, `k6_combo` (gyro LCP) | fall in 2-10 s, forward or in a growing fore-aft rock; k6's first step catches the rug |

### 28.1 What the logs show

1. **The engage lurch.** Without `--ready-act` every policy, h4 included, pitched from -5° (the ready pose) to +10-12°
   forward within 0.75 s of engaging. The k policies without the gyro penalty recovered to +1..+6° by 1.5 s. The two
   with it (k2, k6) turned fastest (56-85°/s against 21-69), overshot back and fell or rocked. The o policies were
   blind through it: their inputs froze at the engage frame, and the settle gate held the clock short of the first
   lift-off while the body pitched, so the first fresh sample came after the fall had started. **`--ready-act` (blend
   into the pose the policy itself holds) removes it; o3 started cleanly with it. It is now the runner's default
   (`--no-ready-act` for the old blend).**
2. **Sideways falls are lift-offs taken while the body still rolls outward.** All 22 sideways falls came during a
   swing, tipping over the stance foot. The steps that fell lifted off at +5-7° toward the stance side and still
   rolling that way at 14-28°/s. h4's median step lifts at +0.5°; k5's at +4.6° and 21°/s (p75 35), near the edge
   every step, and it then stands on one leg 0.53-0.72 s (h4 0.34-0.50). So the high step itself is not the problem;
   lifting it at the end of the weight shift, with the shift still moving, is.
3. **The o policies' "strafe" is the safe sideways rhythm.** o1 lifts at 0° roll, mid-sway: the shift's momentum
   carries it ~2° over the stance foot during the swing and back, and it never tipped. The closed-loop policies
   shift, then fight the sway (the twitch), then lift late.
4. **The settle gate leaks.** It should hold a lift-off until the body turns slower than 0.3 rad/s (17°/s), yet
   lift-offs went through at 21-28°/s: either it ran to its 1 s cap, or the policy lifts ahead of the clock. Worth
   measuring in simulation before relying on it.
5. **Open loop removes the twitch; freezing the pitch rate destabilises the slow pitch loop.** Gyro RMS during swing
   12-16°/s for the o policies against 20-25 for the others. o3 kept gravity live but held the pitch rate for up to
   1.2 s: its fore-aft swing grew from ~10° to 16-22° peak-to-peak per 2 s before each fall, where h4's stays at
   8-10° for 100 s. Delayed rate feedback on a pendulum adds energy.
6. **The gyro-only gradient penalty is out.** It halved the gyro gain and the shake, but the gain it removes includes
   the pitch damping: k2 and k6 fell forward here, d6 (0.05) fell forward in round 9 and d6b (0.01) tipped sideways.
   An input penalty can't separate the 5 Hz ring from a slow pitch; the run-time action filter, and not seeing the
   ring at all (held or low-passed inputs), can.
7. **The higher reference transfers**: k5 lifts ~2x h4's height on the robot, k4 1.25x; they shake a little more
   (swing gyro RMS 24-25 against 21).

### 28.2 Next: round 11

The target, in the operator's words: **steps like k5, doesn't fight the safe sideways momentum like the o policies,
keeps the fore-aft rock in check, and manages the sideways roll correctly.** From k5, so:
- k5's reference and weights;
- hold the roll inputs (and the joint speeds, which carry the ring) through a step, keep gravity's pitch part and a
  low-passed (~1 Hz) pitch rate live, so the fore-aft balance stays closed loop without seeing the 5 Hz ring;
- teach the lift-off timing: don't lift while still rolling outward (a tighter gate on the roll rate, or a cost on
  outward roll speed at lift-off / in single support);
- no gyro penalty; train longer than round 10's o runs (they were a third to half trained).

## 29. Round 11: k5 with sideways open loop, a roll gate and looser hips (2026-11-14 19:04-20:19, simulation)

Target (operator): k5's steps, the o policies' indifference to the safe sideways sway, the fore-aft rock kept in
check, the sideways roll managed. Per-run detail: `results/robot11/NOTES.md`.

New:
- `obs_hold_pitch_lpf` (Hz) / `obs_hold_late` (mjx_env, evaluation, policy_stand): with `obs_hold_u`, the roll inputs
  and the joints are held through a step while gravity x / z and a low-passed pitch rate stay live; inputs stay live
  until the clock first passes a hold phase (no blind engage).
- `settle_gate_roll` / `_tau` / `_lpf`: the settle gate also holds a lift-off while |roll - its 0.1 Hz mean + 0.2 s x
  the roll rate (2 Hz)| exceeds the threshold. Runner: `--gate-roll <deg>` on any policy; the log gains `phase`,
  `gate_held`, `roll_pred`.
- eval6 sim `loose` (hip roll k 5 / 1.25, twice the fitted give) and `scripts/robot10_analysis/sim_liftoff.py`.

### 29.1 Learnings

1. **The fitted simulator doesn't reproduce k5's sideways falls; a hip roll with twice the give does.** There k5 lifts
   off while still rolling outward at +20°/s (robot +21; fitted sim -8, already returning) and falls 6/6, while h4
   barely changes (3/6 vs 2/6) - the robot's ranking. The tilt test had put the robot's hip roll at k 3.5-5 (§15);
   the round-8 fit (10, soft 2.5 above 0.15 N m) is stiffer than the robot under a high step.
2. **Don't low-pass the pitch rate below ~3 Hz.** 1 Hz alone took k5 from 1/6 to 6/6 falls (forward) with no other
   change: delayed rate feedback pumps the pitch. p2 never recovered from it.
3. **Too many changes at once don't train.** The sideways hold + the high reference + loose hips (p2, p4) gave 2.5 s
   episodes that PPO didn't climb out of in 20-24M; the loose hip range alone (m2) made k5 worse everywhere in 16M.
4. **The roll gate is harmless at run time** (k5 unchanged in the fitted sim) but doesn't save k5 on loose hips alone.

### 29.2 For the robot and next

Nothing new to test from training. Run-time only: `python policy_stand.py run --run k5_high2 --enable --vx 0.015
--act-lpf 0.7 --gate-roll 6` - and the new log columns show whether the gate held the lift-offs the round-10 logs
said it should have. Next round, one change at a time: k5 on the loose hip range for 40M+ (or widening it over
training); the sideways hold from o3 / k5 on the fitted hips with the pitch rate raw or at 3 Hz; then combine.

## 30. Video on the robot, and round 12: the robot falls left (2026-11-14 21:30-23:35)

A camera now records each robot run (`policy_stand.py --video`: Windows ffmpeg, frames on the log's clock;
`video_frames.py` makes labelled contact sheets; videos are git-ignored and blocked by a local pre-commit hook).
Findings, k5 (details `results/robot11/NOTES.md`, `results/robot12/NOTES.md`):

1. **The encoder-FK "early lift" was not a lift**: both boards stay down through the weight shift; the body leans
   4-5° over the stance leg on give the encoders can't see. The lift-offs come on the clock.
2. **The settle gate can freeze a swing**: in 215251 a foot came up early and the gate held the clock at 0.20 while
   the body rolled away; frozen, the reference never landed the foot and the robot fell toward it. Runner fix: the gate
   guard (default on; `--no-gate-guard` as trained) skips the hold while the roll grows away from centre (> 4°, > 10°/s)
   or a foot is up (knees differ > 12°). `--gate-roll` now measures from the standing roll.
3. **Every fall now goes left**: left landings come with the body 3.3° left (p10 7°), right ones +2°, and the left lean
   grows step by step. The IMU zero is right (0.0° standing); the left hip roll carries ~2x the right's load. A pelvis
   mass 1-2 cm left reproduces it in the fitted sim (k5 2/6 -> 4/6 falls, all left).
4. **Training k5 on a +-2 cm sideways mass range (q1-q3, 12-16M) made it worse**, even centred (4-5/12 vs 2/12).
   Like round 11's looser hips: a big new randomization range degrades a fine-tuned high-step policy. Find the robot's
   actual bias and train narrowly around it.

## 31. Hardware fixes, and k5 as the best walker (2026-11-14 23:50 - 2026-11-15 00:55)

Two hardware faults, found with the video and the logs, were behind most of the recent falls:

1. **Loose hip-yaw servo screws.** Tightening all screws cut the body shake (gyro roll/pitch above 3 Hz, RMS) from
   17.7 to ~14.5 deg/s on k5, and the vibration is now small with the action filter.
2. **A zero error on the left leg.** The left foot stood slightly forward and higher than the right, so the robot
   leaned left standing; every fall went left. Re-zeroed standing with `biped_bench.py trim` (net: left hip pitch
   +1.5 deg, offset 993 -> 969; knee unchanged); see `hardware/leg_p0/README.md` "Maintenance and zeroing".

k5 (`--vx 0.015 --act-lpf 0.7`, with and without `--no-gate-guard`: similar), by batch
(`scripts/robot10_analysis/batch.py`):

| batch | trials | median up | median steps | falls | shake (deg/s) |
|---|---|---|---|---|---|
| before the screws (11-14 21:49-22:39) | 19 | 7.8 s | 4 | 14: 12 left, 2 back | 17.7 |
| screws tightened (23:53-00:17) | 13 | 10.2 s | 9 | 11: all left | 14.3 |
| **screws + re-zero (00:39-00:53)** | 17 | **25.7 s** (max 32.9) | **30** (max 42) | **7: 4 left, 3 right** | 14.8 |

After the re-zero, at landing (encoder FK, pelvis frame): left steps land at roll -1.9 deg median, width 132 mm
(p90 153), reach +2 mm ahead of the stance foot (p90 +17); right steps +0.9 deg, 126 mm (p90 140), +16 mm (p90 +25).
The remaining fault: **in some trials the left step becomes much larger and further out than the right, the robot
drifts left and each left step gets risky** (operator). In the numbers: the left steps' wide tail and their short
forward reach.

**k5 is now the best full-body walker**: slightly less reliable than h4, but its large steps (about twice h4's lift)
make far more progress per step. It replaces h4 as the base for the next rounds. Next: the left/right asymmetry of
its step placement (measure on video with the coloured pads; a lateral foot-placement term or a step-width target
per side), then the open-loop-sideways idea from k5 one change at a time (§29).

## 32. Round 13: push resistance, first exploration (2026-11-15 night and morning)

Goal (operator): a robot that resists a shove standing, stepping when it must, and returns to its standing pose. Then
the same while walking: a predicted tip stops the walk, the robot recovers, stands, and resumes. Overnight, 8 runs from
k5, all simulation-screened, then two standing ones tried on the robot. Full record: `results/robot13/NOTES.md`.

**Tools added (kept):**
- **Hand-like shoves in training** (`push_dv`, `push_dur`, `push_lat`, `push_walk`): a horizontal force on the pelvis
  for 0.08-0.2 s, sized so the robot gains dv m/s. The old `push_vel` and Open Duck's kick change the whole robot's
  speed, the planted feet included.
- **A walk -> stand -> resume supervisor**, the same rule in training (`sup*`) and on the robot
  (`policy_stand --recover self|stance|<run>`, `slender_biped/recovery.py`). It fires when roll or pitch relative to
  the engage attitude, plus 0.1 s x the 3 Hz low-passed rate, passes 12 deg.
- **A frozen walker in the env** (`sup_walker`): a recovery specialist trained on exactly the states k5 hands over.
- **Stepping shaping** (`capture_v`, `capture`): stand terms off while the pelvis moves fast; touchdowns paid for
  travel along the motion.
- **MJX push tests** ([push_eval.py](../scripts/push_eval.py)): stand (one shove fwd / back / left / right at
  0.1-0.4 m/s) and walk (one shove while walking, with or without the supervisor). They report the command motion
  after the shove.
- **Robot analysis:** `scripts/robot13_analysis/` (`detector.py`, `stand_push.py`, `recover_log.py`).

**What the IMU can tell (k5 robot logs, 39 falls):**
- k5's normal steps already reach 10-11 deg predicted roll (p99).
- A 12 / 12 deg stop rule catches every fall, but only ~0.2 s before 18 deg, with ~2.4 needless stops a minute.
- 10 / 10 deg gives 0.4 s more warning at ~14 needless stops a minute.

**Simulation, standing, share up after one shove at 0.2 / 0.25 / 0.3 m/s:**
- k5: 89 / 62 / 19 %.
- Every shove-trained policy: 91-100 / 72-83 / 33-42 %.
- **No policy ever stepped.** The capture reward was never paid in 30M steps.
- The supervisor-trained walkers learned to trip the supervisor and stop walking (4-5 stops a run, ~1 cm walked): a
  reward hack.
- Recovery controllers under k5 added 1-5 points while walking.
- Dropping to the stance pose was much worse than doing nothing (35 % against 62 %).
- Shove training made the walker worse (`r13b`).

**On the robot (11:19-11:26):**
- k5, `r13a_stand` and `r13c2_sup_early` standing: 11 of 11 trials fell (7 left, 4 forward).
- From the first visible tilt to 18 deg took 0.34-2.6 s (median ~0.55 s), enough time for a step (k5 swings in ~0.35 s).
- Over that window no joint command moved more than 2-10 deg, and no foot lifted. The simulator shows the same: they
  learned small planted corrections and nothing else.

**Learnings:**
1. **Pushes plus penalties (and a capture reward) don't teach a step** to a policy that never steps. The
   overnight approach, like rounds 1-3 for walking, had no motion to imitate. The papers that recover with real
   steps on hardware have a model-based step generator or a reference behind the steps; pure reward-shaping results
   stay in simulation (`results/robot13/NOTES.md` has the rundown: Atalante, Capture Steps, Ferigo, Yang, Egle,
   Pankert, Disney BD-X, Open Duck).
2. **A walker at zero command stands, by construction:** k5's reference is static at vx 0 and its stepping rewards
   are off, so it has no stepping to fall back on when shoved standing.
3. **A stop that is safer than walking will be learned as a goal.** A supervisor in training needs a cost per stop.

**Next (standing first):** recovery steps the way walking was taught (§0):
1. A reference library of recovery steps indexed by the step target (dx, dy), with its own clock. A quick weight
   shift, a k5-height swing, landing, then a step back to the stance. Sideways: both a weight-transfer step with the
   loaded foot and a crossover with the free foot, compared in simulation.
2. The capture point (from the IMU's tilt and rate, height 0.34 m) picks when, which foot and where, the same function
   in simulation and in `policy_stand`.
3. The step target goes in k5's command inputs, the recovery clock in its clock input, so training starts from k5.
4. Large `ref_q` / `ref_contact` rewards, penalties on top.
5. **Stage A:** step on command, no shoves: learn to take each step and settle. Robot check: a key in `policy_stand`
   triggers a step.
6. **Stage B:** shoves at full size from the start (Atalante: a ramp converged early), the capture point
   triggering the steps.

## 33. Round 14: recovery steps by imitation (2026-11-15 12:50-14:55, simulation)

The round-13 plan (§32): a reference library of recovery steps imitated with large rewards, its own clock, the capture
point choosing the step. Standing only. Full record: `results/robot14/NOTES.md`.

**Tools:**
- [rec_step.py](../slender_biped/rec_step.py): IK-built tables of shift, swing, land and closing steps, indexed by
  (dx, dy outward, side):
  - `rs1`: 1.8 s per step, ±5 cm forward / back, 0-5 cm out, 2.5 cm lift;
  - `rs2`: 2.8 s per step.
  - The step ends in the stance it started from.
- `capture()`: the capture point from the IMU (h 0.34 m) picks the step.
- mjx_env `rec_*`: while a step runs, the reference is the step's, the command input is (dx, dy, side) and the clock is
  the step's. Steps start at random (stage A) and / or from the capture point (stage B).
- `train.py --reset-cmd`: new meaning for the command inputs (see the bug below).
- policy_stand: the w / s / a / d / x keys command a step; `--rec-auto` lets the capture point trigger them.
- push_eval: a `steps` test.

**Results (sim robot):**

| | k5 | `r14c2_slow` (stage A, rs2, 24M) | `r14e_swing` (+ swing pay) | `r14f_cp` (stage B) |
|---|---|---|---|---|
| up after 12 s of commanded steps | 0 % | **84 %** | 53 % | |
| up after one shove at 0.15 / 0.25 m/s, capture-point steps on | 100 / 62 % (planted) | 81 / 28 % | 59 / 28 % | 73 / 29 % |

**Learnings:**
1. **The commanded step is half learned:** from falling at every step to surviving 84 % of runs, but by barely
   lifting (2-3 mm on average against the reference's 12-25 mm).
2. **Capture-point steps the policy can't yet carry out are worse than standing planted.** The trigger starts a
   shift and a lift during the fall.
3. **A slower step lifts more:** 2.8 s per step lifted 4x more than 1.8 s in the same 12M.
4. **Reusing an input the parent policy never saw vary is a trap.** k5's sideways-command input had std 0 and
   untrained weights, so the step command arrived as a huge random input. The first 40 min of training went into
   that. Re-seed the normalizer and zero the weights of any input whose meaning changes.

**Next:** make the step itself work first:
- build the recovery step from k5's own swing (c_T12HH's joint motion, displaced);
- a slower, bigger weight shift;
- a termination when a scheduled swing foot stays down;
- longer stage A.

Then stage B with the trigger tuned against planted balance.

## 34. Round 14 on the robot: the steps transfer, the closing step and sideways steps don't (2026-11-15 15:30-15:55)

With video. Details and the operator's observations: `results/robot14/NOTES.md`
([steps.py](../scripts/robot14_analysis/steps.py)).

**What worked:**
- **`r14c2_slow` on keys: 17 of 18 commanded steps** (sideways out, forward, back) over 92 s. The feet lift 2-5 cm on
  the robot, more than simulation suggested.
- **`r14f_cp`**: a gentle push from behind sometimes gets a forward step that holds (4 of 10 capture-point forward
  steps).

**What didn't:**
- **`r14e_swing` sideways steps fell 4 of 5**, toward the closing foot, as it lifted it without shifting its weight
  first.
- **Every sideways and backward capture-point step fell.**
  - Sideways, the capture-point rule picks the foot on the falling side, the loaded one, which can't lift.
  - Backward, the reference itself is flawed: the IK weighted board pitch low, so the swinging board hangs toe-down
    9-16 deg (worst on back steps) and lands toe-down.

**Next (round 15), from the operator's notes:**
- **Decisive steps:** further (8-10 cm), higher (4-5 cm), boards flat.
- **No scripted closing step:** hold the new stance with a settle gate, then return by retracting the stepped foot,
  or bringing the other one over, as a separate step once quiet.
- **Sideways: step with the free foot** (crossover or unloading hop).
- **Backward: a dedicated back-step reference.**

## 35. Round 15: decisive steps don't learn in 8M; walk-recover-resume built, not trained (2026-11-15 19:00-21:10)

Details: `results/robot15/NOTES.md`.

**The reference, rs3** (`rec_step.build2`):
- **Bigger steps:** up to 8 cm fore-aft or outward, lifting 5 cm (6 cm back).
- **Loaded-leg-friendly:** only a 1.2 cm weight shift first, so the loaded leg can step out (the operator's "retract
  and step out").
- **Settle before returning:** a hold gate stops the clock after the step until the body is quiet.
- **Return:** fore-aft, the other foot closes up; sideways, the stepped foot retracts.

**Geometry without ankle pitch:**
- With a level pelvis both boards are flat in a split stance only with the thighs symmetric about vertical. Planted
  boards tilt up to 11 deg, and the pelvis must sit at the feet's fore-aft midpoint whenever both are down.
- A lifting board tips toe-down up to 26 deg.

**Results:**
- **`r15a_steps`** (stage A on rs3, from r14c2_slow): 0 % up in the steps test after 8M. With zero action the step
  falls 0.34 s in: without a weight shift the policy must make all the balance itself. Round 14 needed 24M for
  smaller, slower steps.
- **`r15w_walk`** (k5 + rs3 + `rec_walk`): ready but untrained; lost to the RAM hangs.
- **Free foot vs loaded foot** for sideways capture steps (`r14f_cp`): 58 % against 55 %, no difference.

**Machine:**
- Three lanes hung WSL twice (7.9 GB RAM).
- Every round before ran two, but these lanes use 3.7 GB each, so even two ran it down to 561 MB.
  `scripts/mem_guard.sh` now kills the newest lane first.

**Why no working policy** (details in the notes):
- **Too big a jump from what worked, unchecked before training:** size, lift, swing speed and weight shift all
  changed at once.
- **An unavoidable lift-shortfall penalty:** `tip_z` records the planned lift, not the reachable one.
- **8M of the planned training,** after the RAM hangs.

**Next (push resistance, now paused for k5's walking stability):**
1. Grow from rs2 one change at a time, trained 24M+ alone.
2. Then `r15w_walk` alone.

## 36. Round 16: k5's falls start at a lift-off the body isn't ready for (2026-11-15 22:00-23:50)

Back to k5's walking stability. Details: `results/robot16/NOTES.md`
([walk_steps.py](../scripts/robot16_analysis/walk_steps.py)).

**The trials (21:37-21:51, 13 with video):**
- 12 fell: 5 forward (new), 4 left, 3 right. 6.8 % of steps end in a fall; the 00:39 batch was 2.8 %.
- The hardware hasn't changed: stance, zeros, volts and loop time all match 00:39.
- **No build-up.** The body's peak roll per stance holds at ~6 deg until the fatal step (15-21 deg). Each fall is one
  bad step.
- **The bad step lifts off when the body isn't ready:**
  - Sideways falls: leaning +3.3 deg (median, p90 +11) toward the swing foot, where good lift-offs lean 2.6 deg the
    other way. Rolling toward the swing foot at lift-off: 10 % of the next stances roll past 12 deg at 0-10 deg/s,
    67 % beyond 20 deg/s (0-2 % when rolling toward the stance foot).
  - Forward falls: a right landing kicks the pitch forward (+18..+36 deg/s) and the next lift-off goes ahead pitched
    +13..+14 deg.
- **Why the settle gate lets them through:** it holds only on the rotation rate (17 deg/s), never on attitude or
  direction. It held < 0.1 s on 90 % of steps.
- **Landings:** the right board lands rolled 3.5 deg (median); 49 % of right landings are > 5 deg off flat (left 21 %).

**The change: an attitude gate** (`settle_gate_att`, env and `policy_stand --gate-att`). A lift-off also waits while
the predicted lean toward the swing foot > 3 deg or the predicted |pitch| > 6 deg (angle from the attitude when the
clock starts, + 0.2 s x the 2 Hz low-passed rate).

**Walk tests** (`push_eval walk`, 14 s, 16 per direction and dv, % up):

| | robot sim, no shove | robot sim, 0.1 m/s | loose sim (both) |
|---|---|---|---|
| k5 | 91 % | 81 % | 2 % |
| k5 + the gate at run time only | 86 % | 83 % | 0 % |
| `m1_attgate` (k5 + the gate, trained 20M) | 92 % | 89 % | 7 % |
| **`m2_attland`** (+ boards landing flat, `foot_tilt` -10 -> -30, 14M) | **95 %** | **91 %** | **13 %** |
| `m2b_attland` (m2 + 10M) | 83 % | 78 % | **34 %** |

- The gate alone, on a policy not trained with it, does nothing.
- Trained with it, `m1` falls about a third less after a shove and walks further (14-18 cm against 13-14 cm after the
  shove). Each difference is about one standard error.
- **`m2` is best on both simulators**, with 6M less training than `m1`. Its falls in the robot sim are half k5's
  (7 % against 14 %), and in the loose sim it stays up 6x as often (13 % against 2 %).
- **`m2b_attland` (m2 + 10M) is mixed.** In the loose sim it stays up 2.6x as often as m2, but in the robot sim it is
  worse than k5 (80 % against 86 %) and walks less far. More training made it robust to loose hips (which reproduced
  k5's robot falls in round 11) at some cost on the nominal model.
- For the robot: m2 first, then m2b, with k5 in the same session as the baseline.

**Machine:** two lanes grew to ~3.9 GB each and `mem_guard` killed the second. `m2_attland` (the gate + boards landing
flat) ran after `m1`, alone, for 14M.

## 37. Round 17: steps that set down instead of crashing (2026-11-16 02:00-08:52, simulation)

The round-16 robot test (8 trials; m2 looked best, 2 of 3 stopped up): the policies "look like k5; every step comes
crashing down and vibrates the whole body". Details: `results/robot17/NOTES.md`.

**Measured:** the gyro's tick-to-tick change is 3-4x higher in the 0.1 s after every touchdown than mid-stance, on the
robot (k5, m1, m2, m2b alike; [landing.py](../scripts/robot17_analysis/landing.py)) and in the simulator
([gait_eval.py](../scripts/gait_eval.py): k5's boards land at 260 mm/s). On the robot it is mostly roll, plus a yaw
twist.

**Cause: c_T12HH's design** (rebuilt exactly by `shift_gait.py`):
- the weight shift unwinds from lift-off, so the body moves back toward the swinging foot while on one leg;
- even held, a 0.12 rad shift leaves the centre of mass 1 cm inside the stance board's inner edge: with the hip's
  give the body tips toward the swing side (CPU sim: rolling toward the landing foot at ~26 deg/s at contact);
- a 0.45 s swing of 0.75 rad needs ~87 rad/s^2; the servos ramp at 35-60, so the foot arrives still moving;
- double support takes 62 % of each step, time the swing can use.

**New gaits** (`shift_gait.py --set-frac/--hover/--place-frac/--rel0`), same 1.2 s step: a 0.6 s swing that comes down
to 20 % of the lift by 72 %, sets down slowly and reaches its spot first; the shift held until touchdown; lift
0.60 rad; shift 0.15 / 0.18. Plus the soft-touchdown penalty `land_v` -100 below 2 cm. Each lane 20M, one at a time.

**Final ranking** (96 runs per row, 20 s at 0.015 m/s):

| | robot sim up | loose sim up | landing speed | jolt after touchdown (mid-swing) | walked |
|---|---|---|---|---|---|
| k5 | 72 % | 0 % | 259 mm/s | 15.3 (5.4) | 27 cm |
| m2_attland | 91 % | 3 % | 260 | 14.7 (5.7) | 29 cm |
| **`l4_mid`** (c_S5mid: soft swing, held shift 0.12, lift 0.60; from m2) | **98 %** | 28 % | 188 | 12.3 (3.8) | 34 cm |
| **`l7_s15`** (c_S8mid15: the same, shift 0.15; from l4_mid) | 91 % | **36 %** | **180** | 12.2 (3.5) | **39 cm** |
| `l9_s18` (shift 0.18; from l7) | 81 % | 36 % | 184 | **11.8 (3.0)** | 39 cm |

- The soft swing cuts landing speed by 30 %; the held shift and the lower lift bring the jolt down; growing the shift
  from a policy that already walks the soft swing gives the most robust policy (0.18 straight from m2 fell sideways).
- At 0.03 m/s l4_mid / l7_s15 stay up 93 / 88 % and walk 50 cm in 20 s (m2 88 %, 39 cm).
- Didn't help: the landing penalty on k5's gait alone, k5's swing with the wide shift, a doubled jolt or landing
  penalty, a higher hover, a slower weight transfer after touchdown, 20M more on the same recipe.
- l4_mid's lowest 10 % of swings clear only 5 mm (watch the rug); l7 11 mm, l9 25 mm.

**For the robot:** l7_s15, l4_mid, l9_s18, with m2 as the same-session baseline, all `--vx 0.015 --act-lpf 0.7
--video`; then the best at `--vx 0.03`.

**Next:** the body still rolls toward the new foot at ~24 deg/s before it lands (a third less is stopped at contact).

### 37.1 On the robot (2026-11-16 10:18-10:25, 9 trials)

Details in [results/robot17/NOTES.md](../results/robot17/NOTES.md).

- **l9_s18 is the best of the round** and goes on the working list. The operator: "best captures the goal of silent,
  soft, efficient steps", but each step "feels like the robot is tripping a little".
  - 41 s and 48 s trials.
  - Peak landing jolt 30 % below k5 / m2.
- **l7_s15** steps softer than k5 but is less stable.
- **l4_mid** trips after 1-2 steps.
- **m2** is like k5.

What the video and logs show:
- **The trip is in the table.** The last third of c_S9mid18's swing keeps a flat board within ~10 mm of the rug
  (the slow set-down). Its equal hip + knee lift moves the foot on a circle about the hip, so the descent is mostly a
  backward sweep along the floor. The board brushes the pile before it is placed.
- **The round-17 rel0 tables jump** the hip-roll targets 3-4 deg at every step boundary: the release after touchdown
  ran past the end of the step.
- **On l9 the body still rolls into each landing at ~30 deg/s,** and the boards land 3 deg on their inner edge.
- **The right foot swings first.** walk_steps.py, landing.py and round 16's description of its attitude gate had the
  sides swapped. Corrected: round 16's fatal lift-offs leaned toward the *stance* foot. The gate, swapped identically
  in training and on the robot, holds exactly those; it is kept as trained.

## 38. Round 18: continuous walking (2026-11-16 11:05-17:57, simulation)

The goal: steps quiet enough that the robot can take the next one right away, stepping continuously like a human or
a G1 instead of step, stop, step. Details: [results/robot18/NOTES.md](../results/robot18/NOTES.md).

**A human-like reference didn't train.** New `shift_gait.py walk` tables: one smooth periodic sway, constant pelvis
speed, a quintic straight-down swing by IK, no holds or jumps. From l9 they fell 60-96 % in simulation. Taken apart:
- **The straight-down swing alone** (on l9's structure): 29 % up.
- **The smooth structure alone** (with l9's swing): 15 %.
- **The missing piece is swing-leg retraction.** l9's equal hip + knee lift swings the foot ~5 cm ahead and pulls it
  back as it lands. That pull-back stabilises the gait, and near the floor it is also the rug skim.

**One change at a time from l9 worked:**
1. **n7** removes the 4 deg hip-roll jump at every step boundary (a round-17 table bug). Loose sim 42 -> 61 %.
2. **n11** does the retraction while the foot is high, then sets it straight down. The rug skim is gone (the lowest
   point over the last third of the swing is 28 / 15 / 6 mm against l9's 12 / 1 / 0); loose sim 83 %.
3. **n15** removes the stop at the centre of the weight shift. Robot sim 91 %, loose sim 86 %, the gate holding 5 %.
4. **1.1 s steps hold** (n16 86 / 80 %; n19 at 10M 88 / 79 %). **1.0 s steps don't** (15-58 %).

**The cost:** the straight-down landing is faster in the simulator (267 against 184 mm/s). A 30 mm lift softens it
(n18 10M: 228 mm/s, jolt 12.7), with some loss of robustness.

**Also:**
- **Evaluate the starting policy on a new table before training on it.** n7's policy walked the 1.0 s table zero-shot
  (75 %), and 10M of training on it collapsed it (6 %).
- **The gate still matters with loose hips** (n7 without it: 75 -> 48 % in the loose sim).
- **Second fine-tunes often lost robustness.** Their 10M checkpoints were sometimes better (n18, n19).

**For the robot:** n15_s20, n19_s25 (10M), n18_s23 (10M), n7_s12, then l9 as the baseline, all `--vx 0.015
--act-lpf 0.7 --video` (checkpoints with `--params params_0010M.pkl`).

## 39. Round 19: less lean, transfer after the foot is down (2026-11-16 19:25-21:09, simulation)

**Round 18 on the robot** (18:52-19:01, 17 trials): n15 / n18 / n19 / n7 fall every 1.5-5 steps (l9: every 19). They
fall outward over the stance foot. The operator: the step is "done" mid-air and the robot is already leaning; far too
much lean; longer, shorter steps.
- **The joints trail their commands by 80 ms** (knee, hip pitch) **and 120-140 ms** (hip roll), so the transfer starts
  with the foot still in the air.
- **At lift-off n15's legs open ~30 deg** (l9's policy cancels most of its table's stance shift: ~20 deg).
- **The gate held n15's clock 14-31 % of the time on the robot** (5 % in simulation). *Corrected 2026-11-17: 5.6 %; the log's `gate_held` is a tick counter and those numbers averaged it.*
- The simulator rated n15 at 91 %.

**Round 19:**
- **The changes:** shift 0.18 -> 0.13; the transfer held 0.1 s past the scheduled touchdown, then 0.2 s; no centre
  pause.
- **The smaller shift is what helps.** p1 (n15's swing) and p3 (l9's swing) stay up 94-96 % in the robot sim, with
  the legs 20-23 deg apart at lift-off. p2 (only the held transfer) 65 %.
- **p4** relaxes the gate's lean limits (5 / 8 deg): the clock stops 4.4 % of the time.

**For the robot:** p3_bump, p1_s13, p4_gate, then l9 as the baseline
([results/robot19/NOTES.md](../results/robot19/NOTES.md)).

### 39.1 On the robot (2026-11-16 23:49 - 11-17 00:03, 19 trials)

Details in [results/robot19/NOTES.md](../results/robot19/NOTES.md).

- **p1_s13 and p4_gate go on the working list:** even smoother than l9, but slower.
  - p4: 115 steps without a fall; 3 of 4 trials walked 45 s and were stopped upright. The most coherent gait so far.
  - p1: 5 falls in 152 steps, two 50 s walks.
  - Body roll per cycle 9-10 deg (l9 13, k5 12.5); the gate pauses 2.5-3.2 % of the time (l9 4.3 %, n15 5.6 %; corrected 2026-11-17).
- **They are slower because they step what the table asks:** 34 mm strides at `--vx 0.015` (the table: 36 mm),
  14 mm/s. l9 (42 mm) and k5 (54 mm, partly the body tipping forward) overstep. In simulation p1 / p4 hardly lengthen
  their stride at vx 0.025-0.03: the policy flattens the commanded stride.
- **The remaining falls go right:** 10 of the 11 tilts past 10 deg, over the right foot. Every policy walks with a
  1-2 deg right bias.
- **p3 trips:** l9's bump swing, shortened to 0.55 s, drags on the rug (38 % of mid-swing under 10 mm).

**The operator, comparing k5, l9, p1 and p4 side by side:** the use of the hips is the step change. k5 absorbs the
impact of each step and then continues; from l9 on the hips roll continuously and compensate, and the steps run into
each other. p1 / p4 are more stable but slower, p4 the most coherent and the slowest. The direction to double down on:
p4's gait, with longer steps.

## 40. Round 20: a farther p4, and the flat-hip leg (2026-11-17 01:07-09:00, simulation)

Details: [results/robot20/NOTES.md](../results/robot20/NOTES.md). 13 runs from p4, each scored in the robot simulator
at 0.015-0.06 m/s and with loose and slow hips.

**A farther p4.** Pushing p4 on today's leg gained 20-35 % ground and lost stability:
- doubling the progress reward (q1);
- paying for longer strides (q2, q5);
- 1.1 s steps (q3);
- faster commands (q4).

The one that kept it is **q7_v5**: p4's table to 0.06 m/s, then continued at a fixed 0.05-0.06. It holds 98 % up and
39 cm at 0.03 m/s (p4: 90 %, 35 cm), and 81 % with slow hips.

**No policy walks at its command above ~0.025 m/s.** All 13 settle at 17-30 mm/s whatever they are asked.

**The flat-hip leg** (`scripts/make_flathip_model.py`):
- **The layout:** pitch / roll / yaw servos front to back under the hip plate, the yaw servo behind the hip under a
  round butt.
- **Lengths:** thigh 178 mm and shin 144 mm, against 138 / 111 mm, at the same height.
- **Less rotation for the same speed:** the gait swings the hip pitch 29 deg at 0.03 m/s, against 36 deg.
- **Results:**
  - With its centre of mass 10 mm back (the butt), nothing walked. With it rebalanced (the torso ~42 mm forward), the
    leg trained.
  - It was more robust than today's leg in 2 of 3 matched pairs and failed the third.
  - It never walked farther.
  - Its swing inertia about the hip is 1.7x, and it lands harder.
- **On this evidence it is not yet a speed gain.** If built: keep the centre of mass over the hips and the shin and
  foot light.

**For the robot:** q7_v5 at `--vx 0.03` and `0.045`, against p4 at `--vx 0.03`.

### 40.1 On the robot (2026-11-17 10:25-10:31, 6 trials)

Details in [results/robot20/NOTES.md](../results/robot20/NOTES.md).

- **p4 remains the most stable, smooth and continuous walker.** At the higher command: 0 falls in 78 steps, stride
  34 -> 42 mm, 17 mm/s.
- **q7 trips on its first steps:**
  - Trained at one high command, it cancels a third of the weight shift: legs 7.6 deg apart at lift-off against p4's
    17.
  - The body pitches forward over the left landing.
  - The simulator rated it the round's best (98 %).
- **q2 looks like k5:** less shift, the body pitching forward, and no faster than p4.
- **Why the flat hip isn't faster in simulation:** the tables hold the pelvis still for 60 % of the cycle and surge
  it during each swing at up to 5x the average speed. The ankle-less boards can't follow, the body rocks back about
  40 % of every surge, and both legs cap at 17-30 mm/s. Next: a constant pelvis speed with p4's shift.
- **Correction:** the gate-hold shares reported from robot logs in §38-39 averaged a tick counter. Held time is l9
  4.3 %, n15 5.6 %, p1 2.5 %, p4 3.2 %.

## 41. Round 21: a continuous gait, the weight shift, small balance losses (2026-11-17 11:36-17:14, simulation)

Details: [results/robot21/NOTES.md](../results/robot21/NOTES.md).

**How p4 falls** (robot, 5 trials, 3 falls). The falls come within one step of a lift-off where the body was 2-4 deg
off its usual roll. The table's weight shift is the same amount and timing whatever the body is doing, and the
policy's correction then overshoots. On average p4 corrects small errors; the falls are the tail.

**A continuous gait** (`shift_gait.py --glide`): p4's table with the pelvis at a constant speed (feet on the floor
sweep back all cycle).
- At p4's shift (w1, x1) it walks 15-30 % farther than p4 in simulation.
- It is less robust: 75-88 % up against 85-94 %, and 80 % through shoves against 95 %.

**The shift amount.** Smaller tables (0.10, 0.08) gave policies that shift later: they lift off with 67-79 % of their
peak spread.
- A new check (`gait_eval.py`: legs apart at lift-off) ranks the robot-tested policies as the robot did: p4 91 %
  (works), q2 86 % (shaky), q7 76 % (fell).
- The simulator scores the late shifters highly, so it can't choose the amount.

**A late-shift penalty** (`late_shift`, at each lift-off): x1 shifts on time (89 %), but only where the table shifts
enough.

**Small shoves in training:** more robust, less distance (x3).

**Leg v2 on the continuous gait (f6):** the most robust policy of the round, and the slowest.

**For the robot:** x1_g13L, x3_g13Ls, x2_g10L at `--vx 0.025`, against p4 at `--vx 0.03`.

### 41.1 On the robot (2026-11-18 00:07-00:24, 26 trials)

Details in [results/robot21/NOTES.md](../results/robot21/NOTES.md).

- **x1_g13L steps on time and butter-smooth, then falls**: 12 of 16 trials, median 9 steps, 10 of them over the left
  foot. Its shift is lopsided (the legs open 17-19 deg on right swings, 12-13 on left), the body droops a little
  further toward the swing side every right step, and the correction finally throws it over the left foot.
- **x2_g10L** is the most reliable of the three and walks like l9; nothing new.
- **x3_g13Ls** reacts to small flexes and twitches; 4 of 4 trials fell on the third step.
- **Speed is stride.** Every policy steps every 1.2-1.3 s. k5 53 mm, l9 49, x1 45, p4 40 (22 / 20 / 19 / 16 mm/s).
- **The operator:** p4's sway is too much; smooth steps come from step timing, not big sways; the p4 / x direction
  gives up too much speed and the sway itself costs stability. Back to k5, with k5's speed and step size as a hard
  requirement (§42).

