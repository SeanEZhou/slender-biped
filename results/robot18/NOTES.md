# Round 18: continuous walking, steps that don't need to settle (2026-11-16 11:05-17:57, 6.9 h of the 8 h GPU)

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

## Results (simulation; gait_eval n48 at 0.015 m/s, 20 s; push_eval walk n16 per cell)

| run | table | robot sim up | loose sim up | walked | landing p50 | jolt after td (mid-swing) | clearance p50 | gate holding | walk test robot / loose |
|---|---|---|---|---|---|---|---|---|---|
| l9_s18 (start) | c_S9mid18 | 75 % (n32) | 36 % | 40 cm | 186 | 11.8 (3.0) | 35 | 8.1 % | 80 / 45 % |
| n1_w12 | c_W1: periodic sway | 40 % | 23 % | 50 cm | 264 | 17.3 (2.9) | 33 | 12.6 % | 44 / 23 % |
| n2_w10 | c_W2: the same, 1 step/s | 4 % | 4 % | 52 cm | 227 | 12.5 (3.1) | 33 | 18.4 % | 7 / 5 % |
| n5_w12d | c_W5: transfer in double support | 4 % | 0 % | 45 cm | 283 | 15.9 (2.8) | 35 | 15.0 % | 7 / 2 % |
| **n7_s12** | **c_S12fix: l9's table, release ends with the step** | **92 %** | **75 %** | 32 cm | 191 | 14.1 (2.5) | 39 | 6.1 % | **84 / 66 %** |
| n8_s13ik | c_S13ik: c_S12fix + the straight-down IK swing | 29 % | 8 % | 48 cm | 234 | 12.0 (2.6) | 34 | 10.8 % | 45 / 15 % |
| n9_w7bump | c_W7bump: c_W5's structure + l9's bump swing | 15 % | 0 % | 38 cm | 185 | 14.3 (3.1) | 38 | 8.6 % | 21 / 3 % |
| n10_s14 | c_S14fast: n7's table at 1.0 s steps (from n7) | 15 % | 4 % | 40 cm | 205 | 15.5 (3.9) | 38 | 16.6 % | 20 / 4 % |
| **n11_s16r** | **c_S16ikr: c_S12fix, IK swing with the retraction done high, then straight down (from n7)** | **90 %** | **85 %** | 27 cm | 271 | 16.7 (2.5) | 40 (p10 35) | **3.7 %** | **87 / 84 %** |
| n14_s19 | c_S19soft: retraction by 0.7, longer descent (from n11) | 62 % | 46 % | 34 cm | 239 | 13.3 (2.9) | 41 (p10 36) | 6.4 % | 64 / 45 % |
| **n15_s20** | **c_S20ns: n11's table without the centre pause (settle0 0, shift over 0.4 s; from n11)** | **94 %** | **94 %** | 27 cm | 268 | 15.4 (2.4) | 42 (p10 35) | 5.7 % | **91 / 87 %** |
| n16_s21 | c_S21t11: n11's table at 1.1 s steps (from n11) | 90 % | 75 % | 32 cm | 266 | 13.9 (2.9) | 38 (p10 33) | 8.8 % | 83 / 80 % |
| n18_s23 | c_S23low: n15's table with a 30 mm lift (from n15) | 65 % | 46 % | 36 cm | 214 | 13.3 (2.5) | 34 (p10 4) | 15.1 % | 63 / 45 % |
| n17_s24 | c_S24t10: n15's structure at 1.0 s steps (4M from n15) | 58 % | 31 % | 33 cm | 287 | 15.1 (3.0) | 46 (p10 36) | 10.8 % | 63 / 51 % |
| n19_s25 | c_S25t11ns: n15's table at 1.1 s steps (from n15) | 69 % | 62 % | 34 cm | 281 | 16.0 (2.8) | 39 (p10 33) | 8.7 % | 72 / 66 % |

- **The walk tables don't train** from l9, whatever the sway. In the CPU sim, n1 and n5 fall sideways **outward over the
  stance foot** at lift-off or late in the swing, within 1-2 s. (Written first as "toward the swing foot": the CPU
  script's roll, -asin(R[2,1]), is positive leaning left, the opposite of the robot logs' convention. Corrected
  2026-11-16 evening.) They ended training at 1500-2000 reward against l9's 4800; their
  episodes are half as long.
- Zero-shot, l9's own policy stayed up 84 % on c_W1, so the table itself can be balanced. Training on it from l9 lost
  that.
- **Removing only the hip-roll jump (n7) doubles the loose-sim survival** (36 -> 75 %), and the shove test goes from
  45 to 66 %. l9's policy scored on c_S12fix as on its own table from the first evaluation (4731), and kept it.
- To find what breaks the walk tables:
  - **n8_s13ik:** c_S12fix with the walk tables' straight-down swing.
  - **n9_w7bump:** the walk structure (double-support sway, constant pelvis speed) with l9's swing.
- **n10_s14:** n7's structure at 1 step/s (c_S14fast: 1.0 s steps, 0.55 s swing), from n7.

**The straight-down swing is what breaks the walk tables** (n8: c_S12fix with only the swing changed, 29 % up against
n7's 92 %).

l9's bump swing (equal hip + knee lift, the board level) moves the foot on an arc about the hip: ~5 cm ahead by
mid-swing, then ~6 cm back as it lowers. That is swing-leg retraction, known to stabilise legged gaits (Seyfarth et al.
2003). Here it apparently matters more than the straight set-down. Its retraction near the floor is also the rug skim
the operator saw.

**n11_s16r:** c_S16ikr keeps the straight-down end and adds the retraction while the foot is high.
- The IK swing reaches 6 cm ahead mid-swing and retracts by 0.8 of the swing, then sets straight down.
- Foot path relative to the hip: 14 -> 58 -> 4 mm (l9: 18 -> 67 -> 4).
- The lowest point over the last third of the swing: 23 / 14 / 7 / 4 mm (l9: 9 / 6 / 4 / 2).
- Trained from n7.

**The walk structure breaks it as well** (n9: l9's swing on c_W5's double-support sway and constant pelvis speed, 15 %
up). Both halves of the walk table cost stability. l9's structure (ease shift, hold, short release, a 0.05 s settle)
with the jump fixed (n7) is the base the rest of the round builds on.

**1 step/s is too fast, even on n7's structure** (n10: 15 % up, the gate holding 17 % of the time). The training
reward fell from 5463 (n7's policy on the new table) to 3536 by 10M. Next, smaller steps from n7:
- **n12_s17:** 1.1 s steps.
- **n13_s18:** no centre pause (settle0 0; the shift takes 0.4 s).
- **Evaluations only:** n10's 0M / 10M checkpoints, and n7 with the gate off.

**n11: retraction while high, then straight down** is the most robust policy of the round. Loose sim 85 %, shove test
87 / 84 %, the gate holding half as often as l9.
- **It no longer skims.** The lowest board point over the last third of the swing is 28 / 15 / 6 mm (n7: 14 / 4 / 0).
  The 10th-percentile clearance is 35 mm.
- **But it lands at 271 mm/s** (n7: 191). The body sags, so the foot meets the floor ~0.06 s early, during the faster
  part of the drop.
- **From n11:**
  - **n14_s19:** c_S19soft. The retraction is done by 0.7 of the swing, then a longer straight-down descent (0.55 of
    the swing).
  - **n15_s20:** c_S20ns, no centre pause.
  - **n16_s21:** c_S21t11, 1.1 s steps.

Evaluations only ([eval_extra1.txt](eval_extra1.txt), robot sim unless noted):
- **n7's policy on the 1.0 s table without training stays up 75 %.** n10's 10M checkpoint: 6 %; 20M: 15 %.
  Training on the faster cadence ruined a policy that could already walk it. A short fine-tune, or none, may be the
  way to a faster cadence.
- **n7 with the gate off** (the clock never stops): robot sim 92 % (unchanged), loose sim 48 % (with the gate 75 %).
  The gate matters when the hips are loose. n11 needs it only 3.7 % of the time, so it stays.

**n14 (longer descent): landings 12 % slower than n11's (239 mm/s), and the jolt after touchdown falls to 13.3, but it
lost stability** (62 / 46 %). As in round 17, 20M more from a good policy costs robustness. Its 10M checkpoint goes
into the final ranking.

**n15 (no centre pause) is the most robust policy of the round**: 94 % up in both simulators, 91 / 87 % in the shove
test. The weight now moves from one foot to the other in one go (release, then shift, with no stop at the centre).
Landings are still n11's (268 mm/s).
- **n18_s23:** n15's table with the lift 40 -> 30 mm (c_S23low: a 25 % slower drop, knee 51 rad/s^2, within the
  servos' ramp; peak clearance 30 mm).
- **n17_s24:** 1.0 s steps on n15's structure, a 4M fine-tune.

**1.1 s steps hold** (n16: 90 / 75 %, shoves 83 / 80 %; 1.0 s fell to 15 % in n10). The gate holds more often (8.8 %
against n11's 3.7 %).

**The lower lift (n18) lands softer (214 mm/s) but lost stability** (65 / 46 %). Its lowest swings scuff
(10th-percentile clearance 4 mm), and the gate holds 15 % of the time.

**1.0 s steps are still out of reach:** n15 zero-shot 60 % up; 4M of fine-tuning (n17): 58 / 31 %. Last lane:
**n19_s25**, n15's structure (no centre pause) at 1.1 s steps (c_S25t11ns), from n15.

**No centre pause + 1.1 s steps together (n19) didn't add up** (69 / 62 %). Each held on its own (n15 94 / 94 %, n16
90 / 75 %). The second change, trained from an already fine-tuned policy, cost robustness.

## Final ranking ([final_rank.txt](final_rank.txt): 96 walks per row, seed 7, 20 s at 0.015 m/s)

| policy | robot up | loose up | walked | landing p50 (mm/s) | jolt after td (mid-swing) | clearance p10 / p50 (mm) | gate holding | lowest point, last third of the swing (mm) |
|---|---|---|---|---|---|---|---|---|
| l9_s18 (start) | 75 % | 42 % | 38 cm | 184 | 11.9 (3.0) | 22 / 36 | 7.7 % | 12 / 1 / 0 / 0 |
| n7_s12 | 83 % | 61 % | 32 cm | 189 | 13.7 (2.5) | 27 / 38 | 7.0 % | 14 / 4 / 0 / 0 |
| n11_s16r | 88 % | 83 % | 28 cm | 269 | 16.4 (2.5) | 35 / 40 | 4.2 % | 28 / 15 / 6 / 0 |
| n14_s19 | 61 % | 38 % | 33 cm | 239 | 13.5 (2.9) | 36 / 40 | 6.5 % | 22 / 12 / 5 / 0 |
| **n15_s20** | **91 %** | **86 %** | 28 cm | 267 | 15.1 (2.4) | **35 / 43** | **5.4 %** | **30 / 18 / 9 / 0** |
| n15_s20 10M | 60 % | 53 % | 30 cm | 269 | 13.6 (2.5) | 36 / 44 | 10.6 % | 35 / 22 / 12 / 0 |
| n16_s21 (1.1 s) | 86 % | 80 % | 32 cm | 269 | 13.8 (2.9) | 32 / 38 | 8.5 % | 28 / 19 / 10 / 0 |
| n16_s21 10M | 69 % | 53 % | 34 cm | 267 | 14.5 (3.1) | 33 / 41 | 10.6 % | 30 / 21 / 12 / 0 |
| n18_s23 (30 mm) | 41 % | 32 % | 37 cm | 215 | 13.9 (2.6) | 5 / 35 | 16.3 % | 26 / 15 / 7 / 0 |
| **n18_s23 10M** | 80 % | 66 % | 35 cm | **228** | **12.7 (2.5)** | 29 / 38 | 8.6 % | 29 / 17 / 9 / 0 |
| n17_s24 (1.0 s) | 46 % | 30 % | 35 cm | 283 | 15.1 (3.0) | 35 / 46 | 10.7 % | 36 / 21 / 11 / 0 |
| n19_s25 (1.1 s) | 61 % | 58 % | 36 cm | 281 | 16.0 (2.8) | 28 / 39 | 8.0 % | 26 / 17 / 7 / 0 |
| **n19_s25 10M** | **88 %** | **79 %** | 31 cm | 273 | 15.4 (2.7) | 31 / 41 | 9.4 % | 30 / 21 / 12 / 0 |

## Verdict (simulation)

- **n15_s20 is the round's best.**
  - It is the most robust in both simulators (91 / 86 %; l9 75 / 42 %).
  - The swing stays clear of the rug to the end (no skim: 30 / 18 / 9 mm over the last third, l9 12 / 1 / 0).
  - It moves its weight across without a centre stop, and the gate holds the clock least (5.4 %).
  - **The price is harder landings in the simulator** (267 against 184 mm/s). The foot drops the last 15-20 mm
    straight down, where l9's skimmed in.
- **n19_s25 10M** is the same structure at 1.1 s steps (0.91 steps/s against 0.83). It is nearly as robust (88 / 79 %).
- **n18_s23 10M** has the lower 30 mm lift: the softest new swing (228 mm/s, jolt 12.7), less robust (80 / 66 %).
- **n7_s12** is l9 with the jump fix only: l9's soft landings, more robust (83 / 61 %), but it still skims.
- **Not yet:** a continuous human-like gait (smooth sway, constant speed, straight-down swing), and 1 step/s.

## To test on the robot (best first)

All with `--vx 0.015 --act-lpf 0.7 --video`. Each run's config carries its table and gate timing.

1. `python policy_stand.py run --run n15_s20 --enable --vx 0.015 --act-lpf 0.7 --video`
2. `python policy_stand.py run --run n19_s25 --params params_0010M.pkl --enable --vx 0.015 --act-lpf 0.7 --video`
   (1.1 s steps)
3. `python policy_stand.py run --run n18_s23 --params params_0010M.pkl --enable --vx 0.015 --act-lpf 0.7 --video`
   (30 mm lift, the softest new swing)
4. `python policy_stand.py run --run n7_s12 --enable --vx 0.015 --act-lpf 0.7 --video` (l9 + the jump fix)
5. Baseline in the same session: `python policy_stand.py run --run l9_s18 --enable --vx 0.015 --act-lpf 0.7 --video`

What to look for:
- Does the foot still brush the rug at the end of the swing? n15 / n18 / n19 shouldn't.
- Does the body still shake after each touchdown?
- Is there less pausing between steps? There is no stop at the centre of the weight shift now, and the clock holds
  less often.
- Do the straight-down landings sound louder than l9's?

Afterwards:
- `scripts/robot17_analysis/landing.py <logs>`: the jolt after touchdown (l9 20.3 RMS, peak 31).
- `scripts/robot18_analysis/swing_clear.py <logs>`: the swing height, measured.
- `scripts/robot16_analysis/walk_steps.py <logs>`: per step (now reads each run's table timing).
