# Round 28: the robot's falls in the simulator, then trained against (2026-11-20 11:23-17:23)

**User's plan:** reproduce each fall of the Nov 19-20 trials in the simulator, then tune the policy to avoid those
falls while keeping normal walking. 6 h of GPU.

## Tools (scripts/robot28/)

- `falls.py`: classifies the 71 trials (falls.json). 56 ended in a fall. The onset is when the body leaves its
  walking range for good: 8° roll or 12° pitch.
- `robostate.py`: puts the simulator robot (gait_eval's robot settings) into a logged robot state:
  - From the log: joint angles and speeds, the IMU gravity and gyro, the clock phase, the last two low-passed actions
    and the last servo targets.
  - The body's linear velocity isn't logged. It is fitted by shooting: 143 candidates (vx -0.25..0.25, vy -0.3..0.3 m/s),
    keeping the one whose first 0.2 s best follow the logged tilt. Typical error ~1° rms (velocity 0: 2-7°).
  - Checked: the policy's actions, recomputed from the logged inputs, match the logged actions to ~0.01. This also
    identifies the checkpoint: the 11:09 z7_anc2 trials ran 12M, the 10:08-10:17 ones 25M.
- `repro.py`: per fall, 32 randomized sim robots started at several times around the onset, running the logged policy
  with its logged gate setting.
- `traj.py`: sim roll/pitch against the log from one state.
- `build_bank.py`: the reset bank, plus mjx_env `reset_bank` / `reset_bank_prob` / `reset_bank_noise`.

## What reproduces, and what doesn't (z7_anc2 12M, trials 110940 / 110958)

**The tick log of the fall:**
- The left step lands with the body still rolling left: -7.5° at touchdown, -30 to -50°/s.
- After it, the right foot never comes back down. FK puts it 15 mm, then 44 mm, above the left. The robot balances on
  the left foot alone at -11° roll for ~0.6 s (the IMU agrees with the gyro integral, so the lean is real).
- Then the policy runs its normal shift for the right step (hip roll L +6→+2, R +3→+15), lifts the right leg wide
  and high, and the robot topples left.

**From the stuck double support (the state before the terrible step)** the simulator reproduces it:

| trial | start | sim robots falling, side |
|---|---|---|
| 110940 | t 6.9 s, roll -11° | 69 %, left |
| 110958 | t 9.6 s | 100 %, left |
| 100933 | the right lift-off | 100 %, left |
| 101501 | 0.4 s before the onset | 100 %, forward |

**From 0.4 s or more before the onset, the simulator doesn't fall** (0-6 %).
- Started at the left lift-off, the simulator follows the robot's roll until the left touchdown.
- After that the robot's roll goes to -11 and stays; the simulator's peaks at -4 to -5 and comes back.
- The robot's pitch drops from +16 to 0 at the landing; the simulator's stays near +12.
- None of these changes made the simulator lean like that: hip-roll give doubled or x4, very low friction (0.3), a
  soft rug (tc 0.08), a left pelvis mass offset of 8 / 15 / 25 mm.

So the robot's left landing is a sim-to-real gap this round doesn't close. What can be trained is what the policy does
once it is in that state.

## The reset bank and the held-out test

**Bank** (models/bank/robot28.npz):
- From 34 trials of the 1.2 s-step policies. Every third fall (12 trials, results/robot28/holdout.json) is left out
  for testing.
- 381 'bad' states, every 0.1 s from 1.2 s before each fall's onset to 0.3 s before the log ends, repeated 4x.
- 1360 'normal' states from steady walking, every 0.7 s.
- 2884 states in all; median velocity-fit error 1.4°.
- The velocities were fitted with z7_anc2 12M for every state (one compile; the fit covers only 0.2 s, and the logged
  actions and servo targets are used as they are).

**Held-out falls** (`hold_summary.py`): 32 sim robots per start state, the policy gate-free. Share falling:

| policy | from the onset or after | from 0.4 s before | from earlier |
|---|---|---|---|
| z0 (20M) | 76 % | 36 % | 10 % |
| z7_anc2 12M | 71 % | 33 % | 9 % |

## Training (queue: scripts/robot28_queue.sh; 20M each from z7_anc2 12M with its recipe)

| run | change |
|---|---|
| z8_b3 | 30 % of the episodes from the bank, gate in training (as z7_anc2) |
| z8_b3ng | the same, gate-free training |
| z8_ctlng | control: no bank, gate-free |
| z8_b5mir | 50 % bank + mirror_aug, gate-free |

### z8_b3 (13:12): worse everywhere

Gate-free, compared with its starting point z7_anc2 12M:

| | z8_b3 | z7_anc2 12M |
|---|---|---|
| up at 0.015 / 0.025 / 0.035 | 67 / 48 / 52 % | 96 / 94 / 98 % |
| up with loose hips | 17 % | 81 % |
| up with step errors | 33 % | 73 % |
| start/stop: up at the end | 18 of 32 | 28 of 32 |
| held-out falls (onset or after / 0.4 s before / earlier) | 74 / 39 / 13 % | 71 / 33 / 9 % |

- Its first lift-off shift is +14°, and it is leaning further left again (L touchdown roll -10°).
- **Likely cause:** half the bank is states at or after the fall onset. Many of those can't be saved, and 15 % of the
  episodes starting in them hurts the normal walking without teaching a recovery.
- **Queue 2** (scripts/robot28_queue2.sh) replaces z8_b5mir with milder banks, both at 30 % of the episodes:
  - z8_pre3ng (robot28pre): the normal robot states plus the bad states up to the onset only (19 % of the bank);
  - z8_nor3ng (robot28nor): the normal robot states only.

### z8_b3ng (13:52): the same with gate-free training, also worse

| | z8_b3ng |
|---|---|
| up at 0.015 / 0.025 / 0.035 | 62 / 46 / 42 % |
| up with loose hips | 2 % |
| up with step errors | 29 % |
| start/stop: up at the end | 16 of 32 |
| held-out falls (onset or after / 0.4 s before / earlier) | 80 / 38 / 16 % |

- It is symmetric (shift L/R +11.4° / +11.6°, landings +24 / +36 mm).
- The control z8_ctlng (no bank, 14:10) tells whether the bank or the fine-tuning itself is the cause.

### z8_ctlng (14:32): the control drifts too, so fine-tuning from 12M is itself the problem

| | z8_ctlng (no bank) | z8_b3ng (bank) | z7_anc2 12M (start) |
|---|---|---|---|
| up at 0.015 / 0.025 / 0.035 | 90 / 69 / 50 % | 62 / 46 / 42 % | 96 / 94 / 98 % |
| up with loose hips | 0 % | 2 % | 81 % |
| up with step errors | 48 % | 29 % | 73 % |
| start/stop: up at the end | 27 of 32 | 16 of 32 | 28 of 32 |
| held-out falls (onset or after / 0.4 s before / earlier) | 71 / 39 / 10 % | 80 / 38 / 16 % | 71 / 33 / 9 % |

- Restarting PPO from a checkpoint drifts away from it, as in round 25's control. The original z7_anc2 run went on
  from 12M to 25M without losing its up-rate, but a restart begins with fresh optimizer state at the full learning
  rate (3e-4).
- The bank adds more loss on top of that.
- No policy so far improves the held-out fall states.
- **Change:** the last run is z8_nor3lr (normal robot states only, 30 %) at a learning rate of 1e-4.

### z8_pre3ng (15:10): the worst, though its bank is 81 % normal states

| | z8_pre3ng |
|---|---|
| up at 0.015 / 0.025 / 0.035 | 27 / 27 / 21 % |
| up with loose hips | 2 % |
| up with step errors | 8 % |

**The likely culprit is the fitted velocities, not only the bad states.**
- 19 % of the normal states sit at the grid edge, |vy| 0.3 m/s; the median |vy| is 0.1 m/s.
- These are shove-sized sideways speeds, probably where the fit absorbs the sim/robot mismatch. In 30 % of the
  episodes they act as shoves at the start.
- **The last two runs, both at a learning rate of 1e-4:**
  - z8_preclp: the pre-onset bank with the velocities capped at 0.08 m/s;
  - z8_ctllr: no bank, the control for the lower learning rate.
- z8_nor3lr was dropped.

### z8_preclp (15:50): capped velocities and a lower learning rate didn't help

| | z8_preclp |
|---|---|
| up at 0.015 / 0.025 / 0.035 | 23 / 12 / 25 % |
| up with loose hips | 4 % |
| up with step errors | 10 % |
| start/stop: up at the end | 9 of 32 |
| training episode length (eval, steps) | 563 → 348 |

- The logged joint speeds are consistent: they correlate 0.90-0.98 with the finite difference of the logged angles,
  with the same spread. The reset itself looks right.
- So any bank of robot states in training costs normal walking heavily. The likely reason: the robot's states are
  outside what the simulator's gait produces, many such episodes fall early, and PPO pulls the policy toward them.

## Every fall reproduced? (results/robot28/repro_summary.txt; 17:11)

All 56 falls of Nov 19-20, 13 policies, each run in the sim with the policy that ran (its checkpoint identified from
the logged actions, its gate as it ran). 32 randomized sim robots per start; starts from 1.6 s before the onset to
0.6 s after it.

- **47 of 56 reproduced:** more than half the sim robots fall from at least one start. The side isn't checked.
- **15 of those already from 0.4 s or more before the onset.** In the rest, only once the robot is in the state:
  stuck on one foot, or the step already going wrong.
- **Not reproduced (9):**
  - z0 115616, 101817, 101936 (late left falls after 30-40 steps);
  - z6_e 004954;
  - z5_shv10 215434;
  - h4s2 103302;
  - h9s 102718;
  - g7g 102944, 102956.
- **Controls:** 74 starts from steady walking, at least 4 s before any fall. 3.9 % of the walkers fall on average;
  2 of the 74 starts pass 50 %. So the reconstruction doesn't make the sim fall by itself.

## Conclusions

1. **The falls reproduce once the robot is in its bad state; the way into it mostly doesn't.**
   - In the simulator, the logged policy makes the same terrible step from the stuck states and falls.
   - From normal states 0.4 s and more earlier, it mostly walks on. The robot's left landing tips the body (-11°
     roll, the right foot off the floor, pitch +16 → 0), and no sim change tried reproduces that.
   - That landing is where the sim and the robot differ. It is the next thing to model: the foot on the rug, ankle
     or hip give under load, the stance foot slipping.
2. **Training from robot states failed.** Five fine-tunes from z7_anc2 12M, with and without the bank, gated and
   gate-free, at learning rates 3e-4 and 1e-4. Every one lost normal walking (best z8_ctlng without a bank: 90 / 69 /
   50 % up). None improved the held-out fall states (71-80 % still fall from the onset; z7_anc2 12M: 71 %).
   - **Restarting PPO from a checkpoint drifts by itself** (the control); a robot-state bank costs more on top.
   - The fitted velocities include shove-sized ones, but capping them didn't help.
   - Most of these states may not be recoverable for this robot. Once it balances on one foot at -11°, the policy
     would need a step it never learned (e.g. putting the right foot back down at once).
3. **No new robot candidate this round.** z7_anc2 12M stays the best:

   ```
   python policy_stand.py run --run z7_anc2 --params params_0012M.pkl --enable --vx 0.02 --act-lpf 0.7 --video --no-gate
   ```

## Correction (after the round): the reproducing starts are already too late (renders, `render_falls.py`)

The user's point: the robot steps into these states itself; the question is whether the sim does. Rendered 7 falls
(robot pose from the log next to sim walkers from onset -1.6 / -0.8 / -0.4 / 0 / +0.3 s, plus roll/pitch plots;
the videos stay out of the repo). In every one, the starts that reproduce begin once the robot has already left its
normal walking, and from the earlier starts the sim does not follow the robot in:

| fall | the robot | the sim from before that |
|---|---|---|
| z7_anc2 12M 110940 | at -11° on the left foot from +0.1 s, stays there | from -0.4 / 0 it reaches -11° too, then rights itself within 0.2 s; only +0.3 (already stuck) falls (72 %) |
| z7_anc2 12M 110958 | -9° at the onset, stays, then -20° | the same: reaches -9°, back to 0; even +0.3 only 9 % |
| z7_anc2 25M 101501 | pitches forward to +15° from -0.4 s | from -0.8 it rolls right (+9°) instead and walks on; from -0.4 (already pitched) all fall |
| z0 103024 | rolls left to -8° by the onset | from -0.8 / -0.4 it rolls right (+5 to +11°) |
| z7_rgq 100719 | drifts left to -9° by -0.8 s | from -1.6 it stays near 0°; from -0.8 (already at -9°) all fall, faster and pitching forward |

So the round's '47 of 56 reproduced' mostly means: from a state that is already lost, the sim falls too. The robot's
way into those states is not reproduced; the sim's walkers diverge from it within 0.2-0.5 s, often to the other side.
That fits the robot's own left bias (round 27). The thing to model is that divergence, not the fall.

## Next

- **Model the left landing.** Fit the sim to the robot's logged landings (roll, pitch, foot heights after touchdown)
  from the bank states, with `traj.py` as the check. The candidates are rug compliance per foot, ankle-roll give
  under load, and stance-foot slip.
- **A recovery the policy can use from the stuck state:** when only one foot is down and the body is tilted past
  ~8°, hold the clock and put the other foot down. The gate does this only at the lift-offs. Something like the
  round 14 recovery step would also work.
- **Fine-tune without the restart drift:** keep the optimizer state, or use a much lower learning rate with a KL
  limit. Or continue the original run instead of restarting it.
