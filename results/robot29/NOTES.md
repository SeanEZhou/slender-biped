# Round 29: z0 with start and stop, without losing its walking; turning in place (2026-11-22 01:28-)

Operator: round 27's start/stop policies started well but walked worse than z0 on the robot (speed, stability; the
left limp). Tonight: tune z0_y2c to start and stop; with time left, turning left / right in place, then walking on.

## What changes

New training losses (scripts/train.py `install_anchor_mirror`):
- **`--anchor C`**: the squared distance between the policy's mean action and z0's (the start checkpoint's, with its
  own frozen normalizer), weighted by the commanded speed: 0 at vx <= 0.01 m/s, 1 at vx >= 0.02, and 0 while a turn is
  commanded. Walking stays z0's; standing, starting, stopping and turning are free to change. Round 28 found that a
  fine-tune drifts away from its checkpoint even with nothing changed; round 27's fine-tunes drifted into the limp.
- **`--mirror C`**: left/right symmetry loss on the mean action (Mittal et al. 2024): the action for an input against
  the mirror image of the action for the mirrored input (the env's mirror_aug map: joint swap, signs, phase + pi).

The start/stop recipe is round 27's z7_ip (in-place table c_K2ip, walk_state, commands that start and stop), no
re-anchored reference, no staggered starts.

Turning (slender_biped/mjx_env.py, round 29 block at `turn_prob`):
- z0's lineage was trained with a heading command (heading_cmd: the policy sees the corrective yaw rate, up to
  +-0.3 rad/s, toward a heading offset of up to +-23 deg while walking). A turn is a new target heading with vx 0:
  `turn_prob` of the nonzero commands, 29-92 deg either way; with walk_state the robot steps in place while the
  corrective rate is above `turn_go` (0.05 rad/s, ~3 deg off).
- `turn_yaw`: a turning reference. Both hip yaws get the same offset o (the right yaw axis is mirrored, so toe-out on
  one leg, toe-in on the other); during a left swing o moves to +D/2, during a right swing to -D/2, D = scale x rate x
  step time; the stance hip unwinding turns the pelvis by D per step. `turn_yaw_scale` 0.67: up to 0.2 rad/s, hip yaw
  +-7 deg.
- policy_stand.py: keys `l` / `r` turn the target heading by `--turn-deg` (45); the robot turns in place and walks on
  at the speed it had (or stays standing). New log columns wz_cmd, yaw_est, head_t, yaw_ref.
- scripts/robot29/turn_eval.py: walk 5 s, turn in place 10 s (half left, half right), walk 7 s.

## Results (sim robot, 48 walkers x 20 s; start/stop 32 walkers; symmetry limp_sim 24 walkers at 0.02)

Columns: up at 0.015 / 0.025 / 0.035 m/s; walked in 20 s at 0.025; up with loose hips; up with step errors; start/stop
test (start, stop at 8 s, restart at 13 s): up at the end of 32; the weight shift at the first lift-off (table +13.8);
symmetry (limp_sim, gated): each side's landing dx and weight shift before the step (L / R).

| run | from | recipe | gated: up | walked | loose | step err | start/stop | 1st shift | gate-free: up | walked | loose | start/stop | 1st shift | L / R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z0_y2c | | (no start/stop) | 94 / 92 / 96 % | 38 cm | 33 % | 56 % | 31 (no stop) | +3.0 | 96 / 96 / 98 % | 38 cm | 52 % | 32 (no stop) | +1.1 | +23 mm +9.0° / +27 mm +8.9° (dy 114 / 132) |
| z9_a1m | z0 | anchor 1, mirror 1 | 92 / 94 / 92 % | 37.4 cm | 12 % | 79 % | 29 | +13.0 | 92 / 94 / 92 % | 37.3 cm | 31 % | 30 | +11.4 | +24 mm +9.5° / +25 mm +9.5° |
| z9_a5m | z0 | anchor 5, mirror 1 | 94 / 92 / 94 % | 37.1 cm | 19 % | 67 % | 31 | +13.5 | 96 / 96 / 96 % | 37.1 cm | 29 % | 28 | +11.8 | +23 mm +9.6° / +25 mm +9.9° |
| z9_as5 | z0 | anchor 5, mirror 1, ANCW + STAG | 96 / 96 / 96 % | 35.4 cm | 19 % | 69 % | 31 | +15.2 | 98 / 94 / 94 % | 35.5 cm | 25 % | 31 | +12.6 | +23 mm +10.2° / +24 mm +10.1° |
| z9_ng | z0 | anchor 1, mirror 1, trained gate-free | 79 / 65 / 69 % | 41.2 cm | 2 % | 35 % | 23 | +11.8 | 77 / 69 / 69 % | 41.5 cm | 0 % | 23 | +11.8 | +25 mm +10.7° / +27 mm +10.2° |
| z9_t0 | z9_a1m | turns, no turning reference | 94 / 92 / 92 % | 34.6 cm | 38 % | 67 % | 29 | +14.7 | 94 / 90 / 94 % | 35.2 cm | 42 % | 29 | +13.5 | +20 mm +12.4° / +20 mm +12.2° |
| z9_t1 | z9_a1m | turns 30 %, reference 0.67 | 94 / 81 / 88 % | 38.6 cm | 8 % | 58 % | 29 | +13.7 | 88 / 92 / 90 % | 41.6 cm | 8 % | 28 | +11.3 | +23 mm +10.0° / +24 mm +9.5° |
| z9_t2 | z9_a5m | turns 30 %, reference 0.67, anchor 5 | 96 / 88 / 88 % | 39.4 cm | 10 % | 56 % | 29 | +14.0 | 94 / 90 / 90 % | 40.8 cm | 8 % | 28 | +12.5 | +25 mm +10.6° / +26 mm +10.5° |
| z9_t3 | z9_a5m | turns 15 %, reference 0.67, anchor 5 | 90 / 90 / 90 % | 39.9 cm | 25 % | 58 % | 29 | +13.5 | 94 / 90 / 88 % | 42.0 cm | 23 % | 28 | +12.3 | +24 mm +10.1° / +24 mm +10.0° |
| z9_t4 | z9_a5m | turns 30 %, reference 0.5, anchor 5 | 98 / 98 / 92 % | 38.3 cm | 23 % | 65 % | 29 | +14.0 | 96 / 96 / 94 % | 38.6 cm | 29 % | 29 | +12.2 | +21 mm +10.3° / +21 mm +10.4° |
| z9_t5 | z9_as5 | as z9_t2 + ANCW + STAG | 96 / 90 / 81 % | 37.9 cm | 4 % | 58 % | 26 | +15.8 | 96 / 83 / 88 % | 38.5 cm | 4 % | 29 | +13.3 | +24 mm +10.8° / +24 mm +11.1° |

Start/stop runs: z9_a1m, z9_a5m (anchor 1 / 5), z9_as5 (+ round 27's reward-only anchoring and staggered starts),
z9_ng (trained gate-free). Turning runs: z9_t0 (no turning reference), z9_t1 (from z9_a1m), z9_t2 (from z9_a5m),
z9_t3 (from z9_a5m, half the turns), z9_t4 (from z9_a5m, gentler reference), z9_t5 (from z9_as5). The turning runs'
walking numbers are without turn commands.

### Turning (turn_eval.py: walk 5 s at 0.025, turn in place 10 s, walk 7 s; 32 walkers, half left, half right)

Turned = yaw change from the turn command to 1.5 s after the walk resumes (median); up = up at the end.

| run | 45 deg L / R: turned, up | 90 deg L / R: turned, up | 90 deg gate-free L / R | heading error after (90) |
|---|---|---|---|---|
| z9_t0 | 4°, 88% / 0°, 88% | 9°, 94% / 2°, 88% | 8°, 94% / 2°, 94% | 78.6 / 104.4° |
| z9_t1 | 41°, 81% / 38°, 81% | 91°, 81% / 85°, 81% | 96°, 94% / 89°, 81% | 3.1 / 2.1° |
| z9_t2 | 43°, 88% / 45°, 94% | 94°, 94% / 88°, 81% | 94°, 88% / 88°, 88% | 5.4 / 2.4° |
| z9_t3 | 46°, 94% / 46°, 81% | 93°, 88% / 89°, 81% | 96°, 94% / 87°, 94% | 3.8 / 2.3° |
| z9_t4 | 42°, 88% / 43°, 81% | 73°, 94% / 78°, 81% | 82°, 94% / 74°, 88% | 6.3 / 4.7° |
| z9_t5 | 45°, 88% / 46°, 81% | 85°, 88% / 87°, 81% | 93°, 94% / 88°, 88% | 5.1 / 2.6° |

## Conclusions

- **Start/stop from z0 without losing its walking: done in the simulator.** The anchor holds z0's walking: z9_a1m,
  z9_a5m and z9_as5 stay up as often as z0 (92-98 %) and walk as far (35-37 cm against 38), with round 27's start
  (the full weight shift at the first lift-off, +13 to +15 deg; z0 +1 to +3) and stops (28-31 of 32 through start,
  stop, restart).
- **The limp is gone in the simulator.** With the mirror loss every run lands both feet the same distance ahead and
  shifts the same before both steps (z9_a5m +23 / +25 mm, +9.6 / +9.9 deg). Round 27's z7_var: +2.9 / +13.0 deg.
- **What is still lost:** robustness with loose hips (z0 33 % gated / 52 % gate-free; the start/stop runs 12-31 %).
  Step errors are handled as well or better (z0 56 %, z9_a1m 79 %).
- **The steadiest start/stop policy: z9_as5** (round 27's reward-only anchoring and staggered starts on top): 96 %
  at every speed gated, 31 of 32 through start/stop. z9_a5m is the steadiest gate-free (96 % at every speed).
- **Training gate-free fails** (z9_ng: 65-79 % up): train gated, run gate-free, as before.
- **Turning needs the turning reference.** Without it (z9_t0) the policy learned not to turn (0-9 deg of 45-90).
  With it, the policies turn to within a few degrees and walk on in the new direction.
- **Turning costs some walking.** Best balance: **z9_t3** (from z9_a5m, half the turns): 88-94 % up walking, the most
  accurate turns (46 of 45, 87-96 of 90), 81-94 % up through a turn. **z9_t4** (gentler reference) walks best of the
  turning runs (92-98 %) but turns slower (73-82 deg of 90 in 10 s). z9_t5 (from z9_as5) lost walking (81-96 %).

## For the robot (from hardware/leg_p0/software)

Keys: `p` = stop and stand, `g` = walk again; turning runs also `l` / `r` = turn 45 deg left / right in place
(`--turn-deg` to change), then walk on at the speed it had. The turning code in the runner is new and untested on
hardware: check the hip yaws go the right way on the first `l` (the left foot toes out during its swing), and that
the yaw estimate (log column yaw_est, integrated gyro) follows the real heading.

1. Start/stop: `python policy_stand.py run --run z9_as5 --enable --vx 0.02 --act-lpf 0.7 --video W1=front W2=side`
   (gated as trained; or z9_a5m with `--no-gate`)
2. Turning: `python policy_stand.py run --run z9_t3 --enable --vx 0.02 --act-lpf 0.7 --video W1=front W2=side`
   (z9_t4 if z9_t3 walks worse than z9_as5)

## Next

- Loose-hip robustness for the start/stop policies (z0 33-52 %).
- On the robot: whether the symmetric policies fix the left limp, or the robot's own left bias (round 27: about
  20 mm) remains.
- Turning while walking forward (the turns are trained in place; walking commands keep z0's small heading changes).
