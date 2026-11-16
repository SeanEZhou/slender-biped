# Round 14: recovery steps by imitation (2026-11-15 12:50-15:05, ~2.5 h GPU)

The plan from round 13 ([walking_clean.md §32](../../docs/walking_clean.md)): teach recovery steps the way walking
was taught. A reference to imitate with large rewards, its own clock, and the capture point choosing the step. Standing
only.

## The reference: `models/ref/rs1.npz` ([rec_step.py](../../slender_biped/rec_step.py))

- **Grid:** 2 sides (L / R swing foot) x dx -5..+5 cm (5) x outward dy 0..5 cm (3) x 128 phases, over a 1.8 s step.
- **Sequence:**
  - u 0-0.08 (0.14 s): weight shift toward the other foot (3 cm);
  - u 0.08-0.27 (0.34 s): swing foot A by (dx, dy), 2.5 cm lift, land;
  - u 0.27-0.45: the pelvis moves over the new support and toward A;
  - u 0.45-0.64: the closing step, foot B by the same (dx, dy), so the feet are back in the stance layout, displaced;
  - then centre and hold.
- **Ends where it starts:** at u = 0 and 1 the joints are k5's zero-speed stance, so the policy idles on it between
  steps.
- **IK:** damped least squares on the model, pelvis level at the stance height, boards flat (pitch weighted low:
  no ankle pitch). Worst foot error 0.8 mm. A first grid to ±6 cm had a 24 mm IK failure on the 6 + 6 cm diagonal
  (out of reach); ±5 cm is clean.
- **Joint speeds:** peak knee speed in the table ~550 deg/s at lift-off, faster than the servos. The profile
  smooths it and the policy adapts.

## How it runs (mjx_env `rec_*`; policy_stand the same)

- At rest: command 0, clock 0, reference = the stance.
- **A step:** command = (dx, dy outward, side ±1), clock = 2 pi u, reference = the step's.
- **What starts one:**
  - a random step (stage A, `rec_rand` per second after U(`rec_rest_s`) s of rest);
  - or the capture point, when it leaves 3.5 cm (stage B, `rec_cp`).
- **The capture point:** `cp = h (angle + rate / omega)`, h 0.34 m, the tilt from the standing attitude and the 3 Hz
  low-passed rates. Step target = 1.2 x cp, clipped to the grid; side = the side the cp is on.
- **Rewards:** k5's recipe with imitation raised (`ref_q` 3 -> 6, `ref_contact` 3 -> 6, `ref_swing` 6) and the
  walking terms that read the command as a speed off (`progress`, `stride`, `track_lin`, `track_ang`). No settle
  gate, no heading command.

## Runs

| run | lane | what |
|---|---|---|
| `r14a_steps` | 1 | stage A from k5: random steps, no shoves, 24M |
| `r14b_cp` | 1 | stage B from r14a: shoves 0.15-0.4 m/s (full size from the start), capture-point steps, random steps at 0.4/s, 24M |
| `r14ab_mixed` | 2 | both at once from k5, 40M |

Tests ([push_eval.py](../../scripts/push_eval.py), sim robot):
- `steps`: random commanded steps for 12 s, share up and steps per run.
- `stand`: one shove fwd / back / left / right at 0.1-0.4 m/s, with the capture point triggering steps.

Baseline: k5 itself on the steps test (`r14a_steps:params_0000M.pkl`): **0 % up** after 3.5 steps per run on average.
k5 can't do the steps.

## Robot test (when back)

`python policy_stand.py run --run <run> --enable --act-lpf 0.7` (standing, no `--vx`). Keys: w / s / a / d / x = a
4 cm step forward / back / left / right / in place (`--rec-key-step`). `--rec-auto` = capture-point steps (the default
for runs trained with them). Enter / q / space stops. The log has `rec_on`, `rec_u`, `rec_dx`, `rec_dy`, `rec_side`.

Results below as they come in.

## Progress

- 13:20: `r14ab_mixed` (both stages from k5, rs1) at 10-12M: episodes ~1 s (51 control steps), worse than at the
  start (56). Capture-point steps, random steps and shoves from the first second: it never gets to stand. **Stopped**.
- `r14a_steps` at 12M: episodes 113 -> 150 steps (3 s). The swing foot doesn't lift yet (`ref_swing` ~0, the lift
  shortfall `clear` -159): it falls in the first commanded steps.
- 13:26: lane 2 now **`r14a2_slow`**: stage A on **rs2**, a slower step (2.8 s: 0.22 s weight shift, 0.53 s swing,
  about k5's walking pace; shift 3.5 cm), random steps at 1/s after 1-2.5 s rests, 30M. The idea: learn the step
  at a speed k5 already handles, then speed it up.
- **13:28: a bug, both lanes restarted.** k5's input normalizer has std **0** for the sideways command (vy was
  always 0 in its training) and untrained first-layer weights there. So a recovery step's dy (0-5 cm) arrived as a
  huge random input, and the side flag (±1) was ~5 sigma on the turning-rate input. By 12M the normalizer had only
  adapted to std 0.004 (dy still ~6 sigma). New `train.py --reset-cmd s0,s1,s2`: when restoring, the 3 command inputs
  get mean 0 / std (0.03, 0.02, 1.0) and zeroed first-layer weights (policy and critic). The policy starts as k5 at
  rest and learns the step command from there. `r14a_steps` and `r14a2_slow` were stopped.
  - Lane 1: **`r14c_steps`**: stage A on rs1 (1.8 s steps), 24M.
  - Lane 2: **`r14c2_slow`**: stage A on rs2 (2.8 s), 24M.
  - Stage B (shoves + capture point) from the better one, if time allows.
- 13:56: at 12M after the fix, `r14c_steps` (rs1, 1.8 s step) and `r14c2_slow` (rs2, 2.8 s) both doubled their episodes
  (254 -> 525, 268 -> 532 control steps) mostly by keeping the feet down through the scheduled swings: swing pay ~2 %
  (rs1) and ~8 % (rs2) of its maximum. The slow step lifted 4x more, so the fast lane was stopped and replaced by
  **`r14e_swing`**: from r14c2_slow at 12M, swing pay 6 -> 20 and full at 1.2 cm instead of 2.5, lift shortfall
  -2 -> -5, 18M.
- 14:40: **`r14c2_slow` at 24M**: episodes 705 control steps (14 s), ~half still end in a fall; swing pay 82 (~9 % of
  the maximum, ~2 mm average lift). It survives the steps by barely lifting. Stage B **`r14f_cp`** started from it
  (shoves 0.15-0.4, capture-point steps, 14M: the queue had already read the line before it was shortened).
- 14:46: `r14e_swing` at 18M: swing pay 422 -> 718 at weight 20 (~23 % of its maximum, ~3 mm average lift against
  the 12 mm target); ~60 % of the episodes still end in a fall.
- 14:54: `r14f_cp` (stage B from r14c2_slow) at 14M: episodes 402 -> 460 control steps.

## Results (push_eval, sim robot; `eval_stageA.txt`, `eval_r14f_cp.txt`)

**Step on command** (12 s at rest with random steps, ~2.9 steps started per run):

| policy | up at 12 s | touchdowns per run |
|---|---|---|
| k5 (r14a_steps at 0M) | 0 % | ~0 |
| `r14c2_slow` 24M | **84 %** | 5.2 |
| `r14e_swing` 18M | 53 % | 4.6 |

**One shove standing, capture-point steps on** (share up at 0.15 / 0.25 / 0.35 m/s; k5 planted: 100 / 62 / ~10):

| policy | 0.15 | 0.25 | 0.35 | steps started per run |
|---|---|---|---|---|
| `r14c2_slow` | 81 % | 28 % | 9 % | 2.1 |
| `r14e_swing` | 59 % | 28 % | 0 % | 2.4 |
| `r14f_cp` | 73 % | 29 % | 8 % | 2.1 |

- **The step on command is half learned.** In 24M the policy went from falling at every commanded step (k5: 0 %) to
  surviving 84 % of runs with ~3 steps each. But it does it by barely lifting (2-3 mm on average against a 12-25 mm
  reference): weight shifts and shuffles that keep the boards nearly down.
- **Capture-point steps made shove recovery worse than standing planted** (28-29 % at 0.25 m/s against k5's 62 %).
  The trigger fires as the shove lands, the reference starts a weight shift and a lift the policy can't yet carry
  out, and the robot falls during it. 14M of stage B didn't change that (r14f_cp ≈ its parent).
- **Pushing the swing pay up (r14e) lifted more but fell more.** The lift is still the bottleneck, the same
  trade-off walking had before the ankle and the c_T12 reference.
- **The step's speed matters:** the 1.8 s step (0.34 s swing) lifted 4x less than the 2.8 s one in the same 12M.
- **The bug that cost the first 40 min:** an input k5 never saw varied (vy, std 0) was reused for the step command.
  Re-seed and zero any input whose meaning changes (`train.py --reset-cmd`).

## Robot tests

`cd hardware/leg_p0/software`, standing (no `--vx`), `--act-lpf 0.7`. Keys while engaged: **w / s / a / d / x** = a
4 cm step forward / back / left / right / in place; Enter / q / space stops.
1. **`python policy_stand.py run --run r14c2_slow --enable --act-lpf 0.7`**: press a / d / w / s one at a time and let it
   settle between. Does it lift the foot at all on the robot, and does it stay up? (sim: 84 %, low steps)
2. Same with `--run r14e_swing`: higher steps in sim, less stable.
3. `--run r14f_cp` (capture-point steps on by default): gentle shoves only. In sim it is worse than k5 planted;
   this checks whether the trigger fires on the robot as in sim (`rec_on` in the log, the STEP lines in the
   terminal).
4. Optional, to see the reference on the robot: `--rec-key-step 0.02` for smaller steps.

## Next

- **Make the step itself work before any shove.** Options:
  - Train stage A longer (walking needed 30M+ per stage).
  - A slower, bigger weight shift in the reference (the c_T12 shift is 0.45 s with the hip and ankle rolls).
  - A termination when a scheduled swing foot stays down (Atalante-style barrier), so not lifting stops being the
    safe option.
  - Start from k5's own lift: build the recovery step from c_T12HH's swing (the same joints over time), only
    displaced, so the policy is asked for a motion it already does on the robot.
- Then stage B, with the capture-point trigger tuned in simulation (threshold, gain) against planted balance.

## Robot, 2026-11-15 15:30-15:55 (with video; `scripts/robot14_analysis/steps.py`)

Operator's observations:
1. `r14c2_slow` is stable, but its steps are small. Under a real shove (or a trip) the robot needs a decisive, high,
   far step, not a polite one.
2. `r14e_swing`'s steps are better sized. Forward steps work. Sideways it does fine putting the first foot out, then
   falls toward the closing foot when it brings that foot over, without first shifting its weight onto the stepped
   foot. Ideas:
   - let it take time to settle before returning to the stance;
   - let it choose how to return: retract the stepped foot (what a person does) or bring the other one over.
3. Backward steps don't really work: it weight-shifts instead of stepping, like a forward step played in reverse.
4. `r14f_cp` (capture-point steps): a gentle slap from behind sometimes gets a forward step that stabilizes. A side
   push gets a reaction but no recovery. A push from the front (falling backward) doesn't work.

The logs (per step: command, lift from encoder FK, tilt per phase, outcome):

| policy | trials | steps | outcome |
|---|---|---|---|
| `r14c2_slow` (keys) | 1 long (92 s) | 18: 12 sideways out, 4 back, 2 forward | **17 fine**; fell forward in the settle of the 4th back step. Lifts 18-38 mm (first foot), 10-58 mm (closing foot) |
| `r14e_swing` (keys) | 4 | 5, all left foot out | **4 fell right**, in the closing step or the settle (roll +15..+19 deg). First-foot lifts 37-54 mm |
| `r14f_cp` (capture point) | 11 | 15 capture-point steps, 1 key | 10 of the 15 were forward steps (dx +5 cm): 4 fine, the rest fell left or forward in the closing step / settle. All 4 sideways and the 1 backward step fell, sideways ones during the first swing |

- **The steps transfer.** On the robot the feet lift 2-5 cm (encoder FK; the give hides a few mm), far more than the
  2-3 mm the simulator suggested. The slow step is reliable: 17 of 18 commanded steps.
- **The closing step is where they fall.** r14e falls toward the closing foot (roll toward it as it lifts), as the
  operator saw. The reference moves the pelvis toward the stepped foot over 0.27-0.45 of the cycle; on the robot that
  shift doesn't happen in time, and the closing foot lifts with the weight still on it.
- **Sideways capture-point steps lift the loaded foot.** Falling left, the capture point is to the left, and the rule
  picks the left foot. It can't come up: the robot is falling onto it (video: it tips left through u 0.1-0.25, the left
  board still down). The rule has to step with the free foot (a crossover) or unload first.
- **The capture point fires mostly forward** (10 of 15). That fits the robot's forward standing lean, and the
  operator's "a slap from behind gets a step".
- **The back-step reference is flawed.** With board pitch weighted low in the IK, the swinging board hangs **toe-down
  9-16 deg** mid-swing (16 deg for a 5 cm back step) and lands toe-down 7 deg. Forward steps land toe-up 10 deg. A
  toe-down board swung backward drags its toe: one reason the back steps don't work. The walking reference lifts
  boards flat (equal hip and knee bends).

## Next (round 15 proposals)

1. **A recovery library v2 ("decisive steps"):**
   - steps to ±8-10 cm, 4-5 cm lift, boards flat through the swing (pitch weighted high, or the walking reference's
     lift kinematics), a lower pelvis during the step for reach;
   - **no scripted closing step:** step, land, then hold the new stance, with a settle gate. The return to the stance
     is a second, separate step, taken only once the body is quiet (the settle-gate idea from walking);
   - **retract the stepped foot by default** (the human way); bringing the other foot over as a second option,
     command-selected, so the policy or a rule can pick.
2. **Sideways: step with the free foot.** For a fall to the left, a crossover of the right foot in front, or a
   quick unloading hop. Generate both, compare in simulation.
3. **Backward: a dedicated back-step reference.** A flat board, more lift, the hip extension first, a slight
   backward pelvis lean; screen it open loop in simulation before training.
4. **Fire the capture point from a lean that matches the robot:** measure the standing attitude at engage, as the
   runner does now.
