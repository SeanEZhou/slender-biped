# Round 15: decisive recovery steps, and walk-recover-resume (2026-11-15 19:00-21:10, ~2 h GPU)

The plan from the round-14 robot tests ([walking_clean.md §34](../../docs/walking_clean.md)), with the operator's
notes:
- **Big steps:** a tripping person takes a dramatic step, not a polite one, so the steps are further and higher.
- **No scripted closing step:** the robot settles first, then returns.
- **Sideways:** no crossover step. Instead:
  - push back with the loaded foot;
  - quickly unload the loaded foot and step it far out;
  - or, if the other foot is already up, swing it out as a lever.
- **A real back step.**
- **Later:** turn a fall during k5's walk into a recovery and resume walking.

**Outcome:**
- **Two simulator hangs (§ Machine).** WSL ran out of RAM twice, which lost an hour of the budget.
- **The decisive-step policy didn't learn in its 8M steps.** In the simulator it fell in every test run (0 % up).
- **The walking lane never trained.** Its code is ready and tested.
- **Nothing new from this round is worth putting on the robot.** One runner option can be tried with round-14
  policies: `--rec-side free`.

## Why the round produced no working policy

1. **Lost time:** three lanes ran WSL out of RAM and cost an hour. `r15a_steps` got 8M of its 16-22M; `r15w_walk`
   none.
2. **A reference too far from what worked, not checked before training.** Even 16-22M would likely not have helped:
   over the 8M, eval episodes stayed ~4 s (199 -> 204 steps), every one ending in a fall.
   - **rs3 changed everything at once against rs2:** step 5 -> 8 cm, lift 2.5 -> 5-6 cm, swing 0.53 -> 0.36 s, weight
     shift 3.5 -> 1.2 cm, a new return and a gate.
     - The faster swing went against round 14's own finding (the slower table lifted 4x more).
     - The near-zero weight shift leaves all the balance to the policy from the first moment of every step.
   - **The warning was ignored.** The zero-action check fell 0.34 s into a step, and there was no rs2 baseline to
     judge it against.
   - **The swing-height reference (`tip_z`) is the planned lift (5-6 cm), not what the IK reaches (3.5-4.5 cm with
     the boards near flat).** So the lift-shortfall penalty charges even perfect imitation: `clear` -540 per episode,
     one of the two largest terms.
3. **Too much scope:** three lanes (steps, walking, shoves) on one untested reference.

**Next time:**
- Store the achieved lift.
- Run a round-14 policy on a new table before training it.
- Grow from rs2 one change at a time.
- One lane, 24M+, and stop if episode length isn't climbing by ~4M.

## The reference: `models/ref/rs3.npz` ([rec_step.py](../../slender_biped/rec_step.py) `plan2` / `build2`)

- **Grid:** 2 sides x dx -8..+8 cm (5) x outward dy 0..8 cm (3) x 192 phases, over 2.16 s.
- **Sequence:**

  | step | what | time |
  |---|---|---|
  | 1 | weight shift of only 1.2 cm | 0.08 s |
  | 2 | foot A swings by (dx, dy) | 0.36 s |
  | 3 | pelvis to the middle of the new support | |
  | 4 | **hold gate**: the clock stops at u 0.352 until the body is quiet (tilt < 3.4 deg, rate < 17 deg/s), at most 1.5 s | |
  | 5 | return | |

  - The small shift in step 1 lets the loaded leg step out.
  - Step 2 lifts 5 cm (6 cm stepping back); the foot lifts before it travels and lands straight down.
- **The return depends on direction:**
  - Fore-aft, the other foot closes up, as people do after a forward or back step. The stance ends displaced by dx.
  - Sideways, the stepped foot retracts.
  - Each return swing lifts in proportion to its distance, so a pure side step has no closing step.
- **Kinematics without ankle pitch (found building it):**
  - With a level pelvis, both boards can be flat in a split stance only if the thighs are symmetric about vertical.
    So planted boards tilt up to 11 deg in the split stances; the walking reference accepts the same.
  - **The pelvis must stay at the feet's fore-aft midpoint while both are planted.** A first version that moved the
    pelvis home while the stepped foot was still 8 cm away needed a foot 1.5-10 cm into the floor.
  - A lifted board tips toe-down up to 26 deg (lifting by knee flexion), so the lifts are set higher to keep the toe
    clear.
  - IK pitch weight: 1.0 planted, 1.5 swinging, blended by foot height below 2 cm (a step change made 19 deg target
    jumps). Worst foot error 18 mm; largest per-bin joint step 5.7 deg.

## Runs

| run | from | what | result |
|---|---|---|---|
| `r15a_steps` | `r14c2_slow` 24M | stage A on rs3: random steps at rest, imitation 6 / 6 / 20 (`ref_swing` paid fully at 3 cm), 8M (planned 16-22M) | **steps test 0 % up** in every direction (r14c2_slow on rs2: 84 %); eval episodes ~4 s, every one ends in a fall; the lift shortfall (`clear` -540) and tilt (`tilt_lim` -400) dominate |
| `r15w_walk` | `k5_high2` 16M | walk, recover, resume (`rec_walk`): rs3 steps from the capture point (8 cm walking, 3.5 cm standing) and at random, shoves while walking; command weights rescaled (`--reset-cmd-keep`) | never trained: lost to the first hang, then killed by the memory guard at 21:00 before its first checkpoint |
| `r15b_push` | `r14f_cp` 14M | standing shoves (half sideways) on rs3 | not run (dropped for RAM) |

Eval files: [eval_r15a_steps.txt](eval_r15a_steps.txt), [eval_free_foot.txt](eval_free_foot.txt).

**Free foot vs loaded foot (simulation, `r14f_cp`, 8 per cell):** sideways capture-point steps with the unloaded foot
(`rec_cp_free`) instead of the loaded one: 58 % up against 55 % at 0.15 / 0.25 m/s. No difference at this sample
size.

## New code (all tested)

- **`mjx_env`:**
  - the rs3 hold gate (`rec_gate_max`, `rec_quiet_ang`, `rec_quiet_w`);
  - `rec_walk`: the gait between steps, the walking terms off during a step, and after a step the clock restarts at
    0, vx ramps up from 0 and the heading resets;
  - `rec_rand_walk`, `rec_cp_th_walk`, `rec_cp_free`.
- **`train.py --reset-cmd-keep`:** rescale the command inputs' first-layer weights to new normalizer stds instead of
  zeroing them, so the restored policy acts the same on the commands it knew.
- **`policy_stand.py`** (mock-robot tested: key steps, the gate holding 0.4 s through a lean, walk -> step -> resume):
  - the rs3 gate;
  - walk-recover-resume for `rec_walk` runs (with `--vx`, `--rec-th-walk`);
  - `--rec-side free`;
  - key steps default to 3/4 of the table's range (6 cm on rs3).
- **`push_eval.py`:** `--vx`; for `rec_walk` runs, the walk test counts capture-point steps.

## Machine

- **19:39:** three lanes compiling at once (7.9 GB WSL RAM) hung WSL.
- **19:48:** two lanes training plus a third compiling hit kernel page-allocation failures, and the VM hung until a
  restart at 20:41.
- **Every round before this one ran two lanes.** These rs3 / rec_walk lanes use 3.6-3.7 GB each (round 14:
  2.5-3 GB), so even two ran the RAM down to 561 MB at 21:00.
- **`scripts/mem_guard.sh`** (kills the newest train.py below 700 MB available) caught that last one without a hang.

## Robot

Nothing new from round 15 to test. Optional, with round-14 policies:
- `python policy_stand.py run --run r14f_cp --enable --act-lpf 0.7 --rec-side free`: sideways pushes step the unloaded
  foot out (the lever) instead of the loaded one. No better in simulation; cheap to try.

## Next

1. **Get rs3 learnable before going bigger.** Pick one; each alone, no other lane:
   - a curriculum from rs2's timing and size toward rs3 (interpolate the tables, grow with survival);
   - or a middle library (6 cm, 4 cm lift, 0.5 s swing);
   - and at least 24M steps.
2. **Then `r15w_walk`, alone** (3.7 GB): the walking-recovery env and runner are ready.
