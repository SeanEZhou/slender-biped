# Round 10 (2026-11-14 13:13-): single changes from h4_t05, higher steps, open loop within a step

Recipe: h4's own (scripts/robot6_queue6.sh) with settle_gate_max 1.0 and ref_sigma_q 0.1 set explicitly (defaults
changed since). All from runs/h4_t05/params_0030M.pkl. Queue: scripts/robot10_queue.sh (settings in
robot10_common.sh); screen: scripts/robot10_analysis/screen10.sh (eval6, 6 x 30 s, vx 0.02, run-time act_lpf 0.6).

| run | change | steps |
|---|---|---|
| k1_speed | stride_cmd (step target = cmd / step rate) | 16M |
| k2_glcp | gyro-only LCP 0.01 | 16M |
| k4_high | c_T12H: lift 0.60 rad (first-step clearance open loop 39 mm vs 27.5) | 16M |
| o1_hold4 | sensors sampled at lift-offs and touchdowns, held between | 24M |
| k6_combo | k1 + k2 + k4 + dr_com_x | 16M |
| o3_hold4g | o1 with gravity live | 24M |
| k3_com | dr_com_x [-0.005, 0.025] | 16M |
| o2_hold2 | sensors sampled at lift-offs only | 20M |
| k5_high2 | c_T12HH: lift 0.75 rad (59 mm) | 16M |

New references: models/ref/c_T12H.npz, c_T12HH.npz (shift_gait step, c_T12's timing, --lift-knee/--lift-hip
0.60 / 0.75, --swing-h 0.022 / 0.030). Every c_T12 variant falls open loop on the full body (c_T12 too), so the
open-loop test doesn't rank them.

## Results (vx 0.02, act_lpf 0.6)

Baseline h4_t05: fitted+COM 0/6, 16.4 cm, shake 7.3 | r7+COM 2/6, 25.5 cm, 9.7 | r7hard 4/6 | gains 0.047 / 0.096.

Columns: falls, cm per 30 s, 4-8 Hz shake (deg/s); fitted = h4's simulator + pelvis mass 1.5 cm forward; r7 = round-7
ring + mass; hard = r7hard falls; gains hip / knee (deg per deg/s on h4's robot log).

| run | fitted | r7 | hard | gains | note |
|---|---|---|---|---|---|
| h4_t05 (vx 0.02) | 0/6, 16.4, 7.3 | 2/6, 25.5, 9.7 | 4/6 | 0.047 / 0.096 | baseline |
| h4_t05 (vx 0.03) | 0/6, 12.2, 7.1 | 2/6, 25.3, 10.1 | | | |
| k1_speed (0.02) | 0/6, 14.0, 6.4 | 3/6, 20.2, 10.4 | 4/6 | 0.046 / 0.100 | not faster |
| k1_speed (0.03) | 0/6, 6.3, 5.6 | 3/6, 23.1, 10.2 | | | not faster: the per-command target alone doesn't buy speed at h4's pay (round 9's speed came with 2-4x pay) |
| k2_glcp (0.02) | 0/6, 14.1, **5.2** | 3/6, 19.3, 9.9 | 5/6 | **0.025 / 0.051** | gyro gain halved, shake -30% on fitted, distance about kept |
| k4_high (0.02) | 1/6, **28.3**, 8.8 | 4/6, 30.0, 10.4 | 6/6 | 0.049 / 0.102 | longer steps (reach 12-18 mm vs 7-10), but roll p95 10.3 vs 5.9 and more falls; still adapting at 16M (reward 4643 -> 6293, len 680 vs ~770) |
| o1_hold4 (0.02) | 2/6, 11.2, 7.4 | 4/6, 6.6, 9.3 | 4/6, shake **10.7** (h4 16.5) | (0.024 / 0.039 static; held between events, so ~0 at 5 Hz) | still learning at 24M (reward 642 -> 3528, len 142 -> 428): leans forward (walk -6.8 vs -3.8), posture answer halved (knee-hip -0.30 vs -0.57), falls forward; but the least shake on the hard ring so far |
| **k6_combo** (0.02) | **0/6, 46.1**, 7.5 | 5/6, 23.7, 11.2 | 6/6 | 0.031 / 0.055 | 2.8x h4's distance with no falls on the fitted sim, steps 19-21 mm, walks 2.6 deg forward (h4 3.8), gyro gain -35..-43%; but falls on both ring screens (h4 2/6, 4/6); reward still rising at 16M (len 599) |
| o3_hold4g (0.02) | 3/6, 1.4, 11.7 | 4/6, 14.8, 11.5 | 5/6 | 0.032 / 0.041 static | worse than o1: live gravity didn't restore the posture answer (knee-hip -0.43); still learning (len 514 at 24M) |
| **k3_com** (0.02) | **0/6, 42.0**, 7.5 | **0/6, 33.0**, 8.5 | 6/6 | 0.051 / 0.111 | the best ring screen so far; walks 2.5 deg forward. Caveat: the screens put the mass 1.5 cm forward, which k3 trained on, so it is partly tested on its own training range. Much of k6's distance is probably this change |
| k5_high2 (0.02) | 1/6, **48.6**, 7.8 | 6/6, 20.8, 12.0 | 6/6 | 0.047 / 0.103 | c_T12HH: longest reach (16-21 mm) and highest lift, but falls on both ring screens; least adapted of the k runs (len 468 at 16M) |
| o2_hold2 (0.02) | 4/6, 16.1, 9.3 | 6/6, 12.8, 10.5 | 5/6 | 0.036 / 0.039 static | sensors held a whole step: falls forward (walk -7.9), posture answer -0.27; least trained of the three (len 335 at 20M) |

## Foot clearance in simulation (scripts/robot10_analysis/clear.py: fitted + mass 1.5 cm forward, filter 0.6, vx 0.02, 6 x 20 s)

| run | median swing peak | p10 | swings under 10 mm |
|---|---|---|---|
| h4_t05 | 6.7 mm | 3.9 | 75% |
| k3_com | 9.8 | 4.8 | 52% |
| k4_high (c_T12H) | **15.0** | 6.1 | 24% |
| k6_combo (c_T12H) | **16.3** | 6.4 | 19% |
| k5_high2 (c_T12HH) | **33.2** | 5.7 | 14% |
| k2_glcp | 5.9 | 3.9 | 93% |
| o1_hold4 | 4.3 | 3.4 | 91% (35 swings: shuffles) |

h4 lifts a quarter of its reference's 27 mm. The higher reference (39 mm open loop) more than doubles the policy's
real lift: the example gait teaches lifting where the clearance penalty (`clear` -2, in every run) didn't.

## On the robot (2026-11-14 16:43-18:20, rug, cables slack; logs `hardware/leg_p0/software/stand_<run>_20261114_*.csv`)

Operator notes (condensed):
- k6: tilts forward on engage, the first step catches the rug, falls forward; consistent.
- k4 / k5: also tilt forward on engage, but the high steps survive it and use the motion. Best at `--act-lpf 0.7
  --vx 0.015`. **k5 there is close to the next milestone: consistent large steps, limited shaking, until it falls
  sideways.** Each step is still risky.
- k3: not much better than h4 and less stable; the right leg is passive and catches the rug; sometimes continuous
  steps with little waiting.
- k2: strong forward tilt, falls on its face immediately or after a step or two.
- k1: like h4, slightly less stable.
- o1: falls on its face at engage; held through it, **the twitch is practically gone**. Strafes sideways on the loose
  hips, mostly marches in place, never fell.
- o3: needs the same help at the start (none with `--ready-act`), no twitch, walks with h4-like steps, but a
  forward/back rock builds and it falls after ~6 steps.
- o2: too much forward bias; falls on its face at the first step even when helped.
- The o policies never fell sideways: they ignore the harmless sideways strafe that drives the others into fighting it.

Measured (scripts/robot10_analysis/{engage,steps,liftoff,falls}.py; 50 trials + h4's 3 from the morning):

| run | trials | median up | steps | swing peak* | gyro RMS in swing (deg/s) | roll at lift-off toward stance | roll rate at lift-off | falls |
|---|---|---|---|---|---|---|---|---|
| h4_t05 | 3 | 63 s | 264 | 20 mm | 21 | +0.5° | 15°/s | 2 sideways, after 47 / 108 s |
| k1_speed | 1 | 51 s | 70 | 21 | 19 | +2.6 | 16 | sideways |
| k2_glcp | 5 | 3 s | 36 | 19 | 18 | +1.9 | 9 | 5 forward |
| k3_com | 4 | 18 s | 85 | 17 (R leg 15, L 29) | 20 | +3.3 | 15 | 4 sideways |
| k4_high | 7 | 10 s | 92 | 25 | 25 | +2.0 | 17 | 7 sideways (6 lifting L, over the right foot) |
| k5_high2 | 12 | 9 s | 136 | **42** | 24 | **+4.6** | **21** (p75 35) | 8 sideways, 2 fore-aft |
| k6_combo | 8 | 3 s | 28 | 18 | 23 | +1.8 | 11 | all within 3.5 s: a ±8-10° fore-aft rock from engage (1.25 s period), then the first steps |
| o1_hold4 | 3 | (130 s helped) | 175 | 28 | **12** | **0.0** | 24 | 2 forward at engage |
| o2_hold2 | 2 | 6 s | 12 | 23 | 16 | +0.2 | 14 | forward |
| o3_hold4g | 7 | 16 s | 182 | 19 | **15** | +1.5 | 19 | forward/back rock (p-p per 2 s 10 -> 16-22° before each fall) |

\* FK from the joint encoders and the IMU tilt, so the body's tilt is included; compare ratios only.

1. **Engage**: without `--ready-act` every policy, h4 included, pitched from -5° (ready pose) to +10-12° forward in
   0.75 s. h4 / k1 / k3 / k4 / k5 recovered to +1..+6° by 1.5 s. The two gyro-LCP runs (k2, k6) turned fastest
   (56-85°/s against 21-69), overshot back to -3..-8° and then fell forward. The o policies kept going to +23-25° and
   fell at 0.9-1.1 s: their inputs froze at the engage frame, and the settle gate held the clock short of the first
   lift-off while the body pitched, so they were blind through the whole transient. `--ready-act` fixes it (o3
   started cleanly); it is now the runner's default.
2. **Sideways falls are lift-offs taken while still rolling outward.** All 22 sideways falls came during a swing,
   tipping over the stance foot. The step that fell lifted off at +5-7° toward the stance side, still rolling that way
   at 14-28°/s; normal h4 steps lift at +0.5°. k5's median step lifts at +4.6° and 21°/s, near that edge every time,
   then spends 0.53-0.72 s on one leg (h4 0.34-0.50). o1 lifts at 0°, mid-sway: the sway's momentum carries it only
   ~2° over the stance foot and back (its "strafe"), so it never tips. The settle gate (0.3 rad/s = 17°/s) should not
   let lift-offs through at 21-28°/s: either it ran to its 1 s cap or the policy lifts ahead of the clock.
3. **Open loop removes the twitch, but freezing pitch makes the slow pitch loop unstable.** Swing gyro RMS 12-16
   against 20-25. o3 kept gravity live but held the pitch rate up to 1.2 s; its rock grew until it fell. h4's stays
   at 8-10° p-p per 2 s for 100 s.
4. **The higher reference transfers**: k5 lifts ~2x h4 on the robot (k4 1.25x). It shakes more (24 against 21).
