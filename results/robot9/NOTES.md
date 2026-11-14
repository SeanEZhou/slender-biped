# Round 9 (2026-11-14, from 01:50): faster continuous gait from h4, and a breadth-first search on the 5 Hz shake loop

Goal (user): walk faster than h4 while twitching less. Small balancing moves are fine but must be economical; break
the loop where a small vibration makes the policy react and the reaction makes the shake worse. New gait first
(1.5x step length, no wait between steps), mimic it, then one policy per idea against the shake. 8 h of GPU.

## The gait: c_T10L

`python -m slender_biped.shift_gait step --name c_T10L --T-step 1.0 --settle0 0.0 --shift-t 0.40 --swing-t 0.45
--release-t 0.15 --shift 0.12 --lift-knee 0.45 --lift-hip 0.45 --swing-h 0.015 --rise-frac 0.45 --ankle --vmax 0.05 --nv 6`

| | c_T12 (h4) | c_T10L |
|---|---|---|
| step time | 1.2 s: 0.05 settle, 0.45 shift, 0.45 swing, 0.25 release | 1.0 s: 0.40 shift, 0.45 swing, 0.15 release, no settle |
| weight shift | out over swing + 0.25 s, then a pause | out over swing + 0.15 s, straight into the next shift (continuous sway) |
| both feet down | 62% | 55% |
| step at the robot command | 1.8 cm at 0.015 | 2.7 cm at 0.027 (1.5x), 2.7 cm/s |
| training commands | 0.015-0.04 | 0.02-0.04 (2-4 cm steps) |

Mimicking harder than h4: ref_q 3 -> 5 with ref_sigma_q 0.1 -> 0.05, ref_tip 2 (foot heights), progress 1.5 -> 3,
stride paid up to the commanded step length (new `stride_cmd`; h4's was a fixed 5 cm, and h4 walked 1.2 cm/s at any
command). Settle gate kept at the lift-offs (u 0.2 / 0.7) but at most 0.2 s (was 0.5).

## What others did (web search, 2026-11-14)

- Action smoothness losses: CAPS (Mysore et al., arXiv 2012.06644: temporal + spatial terms; the temporal one does most
  of the work), Grad-CAPS (arXiv 2407.04315), LCP (Chen et al., arXiv 2410.11825, gradient penalty; we tried it on all
  sensors as r10: it also cut the posture answer and sagged forward), spectral normalization (arXiv 2504.08246).
  Benchmark (arXiv 2410.16632): hybrids of loss and architecture do best; low-pass filters have memory (lag).
- Quadrotor sim-to-real (arXiv 1903.04628): a software low-pass on the gyro added lag and *caused* oscillation; they
  removed it. So a gyro filter can backfire (our d8).
- Delay randomization made stair-climbing policies smoother (arXiv 2402.06143); "Tune to Learn" (arXiv 2604.02523):
  a lower policy rate lets joints settle before the next command, fewer high-frequency oscillations.
- Microduck / Open Duck (hobby-servo bipeds): backlash modelled as gear play in series with each servo; a 4 Hz
  Butterworth on the actions; a Menlo write-up traced startup oscillation to stale motor-state packets and fixed the IO
  timing, not the rewards.
- "Too Much of a Good Thing" (arXiv 2606.02636): heavy sim2real fidelity / randomization can make policies conservative
  and hurt learning: supports training on a weaker ring (d2) or none (d1) and fixing the rest at run time.
- No source treats a gyro -> policy -> frame-ring loop directly. Our own: the gyro gain on a robot log predicted the
  robot's twitch ranking; the simulated ring did not.

## The branches (all from g1_t10l at 40M, 24M each, one at a time)

| run | idea | what changes |
|---|---|---|
| g1_t10l | base: h4 -> c_T10L on h4's own simulator (no ring) | 40M from h4_t05 |
| d1_ignore | ignore the problem: no ring, run-time filter only | continue g1 |
| d2_ring05 | train on half the round-7 ring: learn walking, not ring-fighting | ring gains 0.5 / 0.35 |
| d3_ring10 | the round-7 ring | 1.0 / 0.7 |
| d4_ring15 | a harsher ring than the robot's | 1.5 / 1.05 |
| d5_hf | filter trained in, but the smoothness penalty on the raw output (round 8: the filter hid the twitch from action_rate) | act_lpf 0.6, act_hf -0.5 (raw output above 2 Hz) |
| d6_glcp | gradient penalty on the gyro inputs only (r10's on everything sagged) | --lcp 0.05 --lcp-idx 3:6 |
| d7_vib | fake 3.5-7 Hz gyro bursts up to 20 deg/s: the body doesn't move, so answering them only costs; teaches "ignore small shakes" | gyro_vib [0.35, 3.5, 7] |
| d8_glpf | 2.5 Hz gyro low-pass trained in (robot: as trained) | gyro_lpf 2.5 |
| d9_blind | no gyro at all: the loop through the gyro is gone; gravity and joint encoders remain | gyro_obs_scale 0 (robot: as trained) |
| d10_land | soft touchdowns: the 12-14 deg/s shake floor comes from the landings | land_v -200 |
| d11_combo | d7 + d5 + d10 | |

d2-d11 except d3 / d4 use the half ring (the user's idea: a weaker ring so the policy learns walking).

## Screen

`scripts/robot9_analysis/screen9.sh <run> [lpf] [vx]`: eval6 6 x 30 s at vx 0.027 on `fitted com_dx=0.015` (h4's
simulator, centre of mass 1.5 cm forward as on the robot), `r7 com_dx=0.015`, `r7hard`; then the gyro gain on h4's
188 s robot log (lower = less twitch on the robot, round 8). Run-time filter 0.6 unless trained with one.

## Results

Baseline h4_t05 + run-time filter 0.6 (same screen): vx 0.015: fitted+com 0/6, 20 cm / 30 s, shake 7.0; r7+com 1/6,
25 cm; r7hard 5/6 (back, at the start). vx 0.027: fitted+com 12 cm, r7+com 3/6. Gyro gain 0.047 / 0.096.

g1_t10l (base, 02:31, 40M): vx 0.027: fitted+com 0/6, **46.5 cm** / 30 s (h4: 12 at the same command, 20 at its own
0.015), shake 7.6; r7+com 3/6, 32 cm, shake 10.5; r7hard 6/6 back in 3 s (starts leaning back +5 deg, like h4).
Gyro gain 0.043 / 0.102 (h4's). At vx 0.035: 44 cm (no faster). Training: progress pay 223 (10M) -> 451 (20M).
Still well short of the 2.7 cm/s command (1.55 cm/s) but 2-4x h4.

d1_ignore (03:00, base + 28M, no ring): fitted+com 0/6, 27 cm, shake 6.8; r7+com 1/6, 25 cm, shake 8.5; r7hard 4/6.
Gain 0.055 / 0.100. Steadier than the base but slower (46 -> 27 cm): more training on the same task traded speed
for staying up.

d2_ring05 (03:28, base + 28M, half ring, old pay): stopped walking: fitted+com 0/6 but 6.5 cm; r7+com 0/6, 6 cm,
shake 6.4; r7hard 4/6. Gain 0.052 / 0.103. The training log shows the same drift in d1 (no ring): progress pay
542 -> 319 (d1), 580 -> 151 (d2); stride 260 -> 150 / 70; ref_swing 319 -> 206 / 80; ref_q and ref_contact up.
Standing in the reference pose out-earns walking, and the ring makes walking costlier (round 7-8's r8 / s2 again).

**03:37 change:** all branches from here: progress 3 -> 6, ref_swing 6 -> 12, ref_contact 3 -> 2. d7_vib restarted
(was 9 min in). Added d1b_ignore and d2b_ring05 (d1 / d2 on the new pay); d1_ignore / d2_ring05 kept.

d1b_ignore (04:06, new pay, no ring): fitted+com 0/6, **57 cm** (1.9 cm/s, 4.7x h4 at this command), shake 7.0;
r7+com 2/6 (forward), 42 cm, shake 9.6; r7hard 6/6. Gain 0.053 / 0.114 (knee 19% above h4's).

d7_vib (04:34, half ring, fake gyro bursts): fitted+com 0/6, 28 cm, shake 8.3; r7+com 1/6 but 7.6 cm, walks -7 deg
forward; r7hard 4/6. Gain 0.059 / 0.100: not lowered. The policy has no memory (obs_hist 0), so it can't tell a fake
5 Hz wobble from a real tilt rate by frequency; it could only lower the gain overall, and didn't.

d5_hf (05:02, half ring, act_lpf 0.6 trained in + raw-output penalty above 2 Hz): fitted+com 0/6, 54 cm, roll p95
5.0 (steadiest roll so far), shake 8.7; r7+com 2/6, 33 cm, roll p95 3.9; r7hard 4/6. But gain 0.093 / 0.119: the hip's
doubled, like s1. Both run behind the same 0.6 filter on the robot, so the raw gains compare directly: worse than h4.
The raw penalty didn't hold the gain (the simulated shake gives it little 5 Hz to see).

d6_glcp (05:30, half ring, gyro-only gradient penalty 0.05): **gyro gain 0.011 / 0.016 (-80% vs h4)**, lowest shake
(fitted 6.0, r7 7.9), but slow (21 cm) and leaning forward (walk -6.2 on r7+com, 4/6 falls forward); its tilt answer
also fell (-0.44 vs -0.57). Penalty too strong: queued d6b_glcp01 (0.01) after the main queue.

d2b_ring05 (05:58, half ring, new pay): keeps walking now but slower: fitted+com 0/6, 30 cm, shake 8.1; r7+com 3/6
forward, 31 cm, walk -6.0; r7hard 4/6. Gain 0.070 / 0.093 (hip up 50%, knee as h4). Training reward fell
(7700 -> 7000 at 14M): the ring costs. vs d1b (no ring, 57 cm): the ring halves the speed without a steadier screen.

d3_ring10 (06:26, round-7 ring): slow everywhere (17-21 cm), falls forward. **On h4's simulator (no ring) it shakes
20.5 deg/s and falls 4/6** although its gyro gain is the lowest of the plain branches (0.050 / 0.086): a policy tuned
to fight one ring shakes itself on a body that rings differently, the round-7/8 robot failure seen in simulation.
The static gyro gain doesn't capture this (the loop's phase depends on the plant).

d9_blind (06:54, half ring, no gyro): relearning balance from gravity + encoders only, 28M wasn't enough: training
episode 190 -> 385 steps (others ~700), screen falls 4/6 on fitted+com (sideways), 5/6 on r7+com; 18-20 cm. Its
residual gain on the (noise-only) gyro input 0.030 / 0.049 (zero on the robot: runs with --gyro-scale 0 from the
config). Needs more training or a curriculum (fade the gyro out) to be a fair test.

d8_glpf (07:22, half ring, 2.5 Hz gyro low-pass trained in): fitted+com 0/6, 23 cm, shake 6.9; r7+com 3/6 forward;
r7hard 5/6. Gain on the filtered input 0.059 / 0.082; the filter passes ~45% at 5 Hz, so its 5 Hz gain is about
0.027 / 0.037 (40% of h4's), with lag. Slow, like the other half-ring branches.

d10_land (07:51, half ring, soft touchdowns): fitted+com 0/6, 22 cm, shake 7.6; r7+com 1/6, 22 cm, shake 8.3 (the
steadiest r7+com of the half-ring branches with d7); r7hard 4/6. Gain 0.064 / 0.108. Shake not below d1b's 7.0 on
fitted; slow like every half-ring branch.

d4_ring15 (08:18, 1.5x ring): the clearest case of d3's pattern. Gyro gain the lowest of the ring sweep (0.032 / 0.058)
yet on h4's simulator **shake 32.5 deg/s** and 3/6 falls forward, 12 cm; r7+com 5/6 forward. Ring sweep in short:
the stronger the training ring, the slower, the more forward-leaning, and the more it shakes on a body that rings
differently (d1b 7.0 -> d2b 8.1 -> d3 20.5 -> d4 32.5 deg/s on the no-ring simulator).

d11_combo (08:46, half ring, fake gyro bursts + act_lpf 0.6 trained in + raw-output penalty + soft touchdowns): the
best-balanced half-ring branch: fitted+com 0/6, 38.5 cm, shake 7.6; r7+com 1/6, 36 cm, shake 8.2, roll p95 4.7;
r7hard 4/6. But gain 0.094 / 0.111 (the filter-trained policies all double the hip gain, as d5 / s1).

d6b_glcp01 (09:15, half ring, gyro-only penalty 0.01): gain 0.028 / 0.049 (about half of h4's), shake 7.7 / 8.1,
but slow (20 / 15 cm) and 1/6 falls (left) on both. The speed loss matches the other half-ring branches, so it is
probably the ring, not the penalty.

d9c_blind (09:43, d9 + 28M, no gyro): still learning (training episodes 395 -> 531 steps); screen 4/6 falls on the
h4 sim (sideways), 3/6 on the ring, 16 / 7 cm; r7hard 6/6. Not ready; the robot will likely tip it sideways.

GPU work ended 09:42 (budget 09:50). 15 runs: g1 + 14 branches (d1, d2 on the old pay).

## Robot trials 2026-11-14 11:04-11:32 (all at vx 0.027 unless noted; h4_t05 with the filter as reference)

`scripts/robot8_analysis/twitch.py` on `hardware/leg_p0/software/stand_*_20261114_*.csv`, 4-8 Hz gyro roll/pitch deg/s, both feet down:

| run | longest s | shake r/p | cmd 4-8 hipR/hipP/knee deg | in bout | end | operator note |
|---|---|---|---|---|---|---|
| h4_t05 | 107.9 | 17.5/14.0 | 0.42/0.65/1.18 | 21% | side | still the best |
| g1_t10l | 12.0 | 19.9/12.7 | 0.36/0.58/0.97 | 15% | side | twitch, side tip |
| d1b_ignore | 17.6 | 20-31/13-23 | 0.5-0.8/0.7-1.0/1.3-2.2 | 35-96% | side | bigger steps, toes catch, heavy twitch |
| d2b_ring05 | 12.0 | 14-23/21-25 | 0.4/0.6-0.9/0.9-1.4 | 29-70% | side/fwd | side unstable |
| d3_ring10 | 4.4 | 11-13/10-13 | 0.6/1.0-1.2/1.4-1.5 | 0-20% | side | side + fwd rocking |
| d4_ring15 | 2.8 | 4.4/5.3 | 0.3/0.3/0.3 | 0% | fwd | side + fwd rocking |
| d5_hf | 7.3 | 16.9/33.9 | 0.64/1.21/1.51 | 90% | side | massive twitch |
| d6_glcp (0.05) | 2.7 | 7-8/4-7 | 0.2/0.3/0.3 | 0% | fwd (pitch -24) | fwd/back rocking |
| d6b_glcp01 | 3.1 | 9-11/6 | 0.2-0.4/0.3/0.4-0.5 | 0-11% | side | no twitch, side tip |
| d7_vib | 33.7 | 16/13.5 | 0.40/0.69/1.16 | 25% | side | few steps, vibration |
| d8_glpf | 11.2 | 12.9/9.4 | 0.30/0.38/0.44 | 0% | side | little twitch, side unstable |
| d9c_blind | 2.7 | 9.4/4.5 | 0.3/0.4/0.3 | 0% | side | immediate side twist |
| d10_land | 13.6 | 17.6/22.8 | 0.49/0.77/1.37 | 61% | side | stuck fighting vibration |
| d11_combo | 32.3 | 27-31/33-38 | 0.6-0.7/1.3/1.7-1.9 | 79-97% | side | massive twitch |

**The sideways fall is the gait, not the twitch.** Nine trials (g1, d4, d6 x2, d6b x3, d7, d9c) end within 3.1 s,
on the 3rd lift-off (~2.2 s after engage, right foot up, roll rising +5 -> +20 over the left foot). The 1st two steps
look like h4's. The difference is after the 2nd landing: h4 lets the roll come back to ~0 (+0.6 deg at 2.5 s) and lifts
again ~0.5 s after landing; the c_T10L policies are still at +3..+7 deg and lift ~0.3 s after landing, so the landing
roll and the new weight shift add up and carry the body past the stance foot. Quiet policies (d6b, d8, d9c) fall the
same way, so it is the shortened double support (release 0.25 -> 0.15 s, settle0 0.05 -> 0, settle gate max
1.0 -> 0.2 s) that the simulator does not punish. h4's gate (hold the clock while |roll/pitch rate| > 0.3 rad/s for up
to 1 s) is likely part of why it survives the robot's ring.

**Twitch on the robot ranks like the simulator said:** gyro LCP (d6, d6b) and blind (d9c) lowest, gyro LPF (d8) next,
then h4 / g1 / d7, then the stronger pay (d1b), then the filter trained in (d5, d11) worst. The LCP gain cut transfers;
d6's 0.05 is too much (no pitch damping: falls forward in 2.6 s), d6b's 0.01 keeps pitch.
