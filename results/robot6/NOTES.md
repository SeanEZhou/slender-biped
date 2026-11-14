# Robot round 6 (2026-11-13 night): the full body on measured masses and give

Plan (user asleep, ~8 h GPU, watchdog every 20 min; don't commit):
1. Model `hw_ank3_b150_m2_10dof` (scripts/make_mass_model.py): legs+pelvis 0.9 kg, torso+head 0.3 kg (was 1.205 / 0.243).
2. Give: `flex_series` (servo damps the shaft, the body can ring), nonlinear (k up to tau0, k2 beyond), at hip roll,
   ankle roll, knee, hip pitch, hip yaw. scripts/tap_sim.py for the sim tap / shift tests (fit inconclusive: hip roll
   alone barely lets the pelvis roll with both feet flat; give at all joints, k 10, tc 6 ms rings at 4.5 Hz z 0.09).
3. tilt_lim reward: roll + backward pitch past 3.4 deg.
4. Queue 1 (scripts/robot6_queue1.sh): g1_mass, g2_series, g3_tilt (from fb2_lean). Then: lean, seeds, legs re-massed.
Eval: gap_fit / fbsim-style in the CPU sim on m2 with the series give, no pull; robot ranking by falls, roll p5-95, steps.

## Log
- 01:47 queue 1 started (g1_mass). 02:08 g1 OOM-killed at 20 M by a 14-process CPU eval (WSL has 7.6 GB): evals now
  run 3 processes (EVAL_PROCS). g1's 20 M params kept.
- g2_series (first try) used an explicit series-give update that was numerically unstable (the old policies fell in
  0.5 s in the evaluator): killed, deleted. Fixed: `series_give` solves the massless shaft implicitly (both envs).
  Check: fb2_lean in the evaluator with the series give (k 10/15, tc 5 ms): 0/3 falls in 10 s, |roll| p95 5-8 deg,
  4-8 Hz shake 6-10 deg/s (old give: 4.6 deg, 6.6). Robot: falls, shake 24-31. Milder than the robot, closer than before.
- 02:13 queue 2 (scripts/robot6_queue2.sh): g2_series, g3_tilt, g4_lean, g5_seed1.
- Legs re-massed model: hw_ank3_b150_legs_pelvis_m2_10dof (0.9 kg).
- g1_mass (20 M), 9 x 30 s, 0.015: fitted [series give] 0/9, |roll| p95 5.3 (fb2 7.4), >4 deg 12% (26%), pitch -2.6,
  but 7.7 cm / 30 s, 6 mm steps (fb2 43 cm, 14-16 mm): the tilt penalty bought calm by stepping in place.
  Old sim: 4.1 vs 5.3 deg, 5 vs 29 cm. -> queue 3 adds forward pay (g6_step).
- g2_series (eval_g2.txt, 9 x 30 s): fitted 3/9 falls (forward), |roll| p95 4.4, fwd -0.3 cm, cmd 4-8 Hz 7.6% (chatter up);
  hard 3/9. g1_mass hard 3/9 (right), 4.8 cm. fb2_lean hard 8/9 (back, 5.8 s): the hard sim (knee -0.09, softer
  series give) breaks fb2 as the robot does. g2 calmer but stalls and falls forward -> queue 3: g6 starts from g1, adds
  g8_oldstep (g1's give + forward pay, tilt -0.5).
- g3_tilt (eval_g3.txt): fitted 3/9 (forward), no steps (fwd -2 cm), cmd 4-8 Hz 9.6%, shake 15.8; hard 2/9. Worse than g2:
  on the series give, the tilt penalty pushes the policy into 4-8 Hz chatter and a forward lean (-3.6 deg) instead of
  steps. -> queue 4: the series give with stronger smoothness (action_rate / action_acc -0.3) and the step pay.
- g4_lean (eval_g4.txt): fitted 3/9 (fwd 2, back 1), no forward progress, cmd 4-8 Hz 10.5%; hard 4/9. Same failure as
  g2/g3: the series-give branch from g2 chatters and stalls. So far g1_mass (round-5 give, re-massed) is the best
  full-body candidate: 0/9 fitted, 3/9 hard, calm, but short steps.
- g5_seed1 (eval_g5.txt; g2's recipe from g1, seed 1): 0/9 fitted AND 0/9 hard, |roll| p95 3.5-3.7, >4 deg 3-4% (the
  steadiest yet) but drifts BACK 15 cm / 30 s with 1-4 mm steps, cmd 4-8 Hz 6-7.5%. Steady but doesn't walk.
- g6_step (eval_g6.txt): fitted 2/9 (fwd), hard 0/9, old 0/9, but no forward progress (-5 to -15 cm). Training logs:
  the tilt penalty (past 3.4 deg, -1) costs ~-1600 / episode vs progress ~+200; ref_swing fell 773 -> 50: the
  policies learned not to step. g1-g6 are all "stand and shuffle" policies. 04:55: stopped g7_step3 (stronger penalty)
  and queues 3-4; queue 5 (scripts/robot6_queue5.sh): tilt past 4.6 deg at -0.2, from fb2_lean: h1_ser, h2_old,
  h3_smooth, l1_mass (legs), h4_seed1.
- h1_ser (eval_h1.txt; fb2 -> m2 + series give, tilt past 4.6 deg -0.2): fitted 0/9, |roll| p95 4.9, cmd 4-8 Hz 3.3% (no
  chatter, as fb2), hard 1/9 (8.2 deg p95); but fwd 0.5 cm: still stepping in place. Training: progress 167 -> -2,
  ref_swing 544 -> 125, tilt -1110 -> -381. 05:35: queue 6 replaces the rest of queue 5 (h2_old keeps running):
  h3_notilt (control, no penalty, more forward pay), l1_mass (legs), h4_t05 (penalty -0.05).
- h2_old (eval_h2.txt; fb2 -> m2, round-5 give, tilt past 4.6 deg -0.2): WALKS. fitted 0/9, 32 cm / 30 s, reach 11.7 /
  11.6 mm (even L/R), |roll| p95 5.5 (fb2 7.4), >4 deg 16% (26%), cmd 4-8 Hz 4.6%. Hard 7/9 (fb2 8/9). Old sim 0/9, 37
  cm. -> the first round-6 robot candidate (fb2's stepping, calmer sideways, on the right masses).
- h3_notilt (control, eval_h3.txt): fitted 3/9 (all left), 38 cm, reach 17 mm, |roll| p95 6.9; hard 9/9 sideways (9.8 s);
  old 2/9. Without the penalty it steps longer and falls sideways: the penalty does buy sideways margin (h2: 0/9 at
  5.5 deg) but at -1 it stops stepping. -0.2 (h2) is a workable middle.
- Legs on the re-massed legs model (0.9 kg), eval_l1.txt / eval_b7_m2.txt: b7_seed2 unchanged: fitted 0/9, 63 cm, reach
  24-25 mm, |roll| p95 7.2; hard 4/9, 27 cm. l1_mass (b7 + series give + tilt -0.2): 0/9 / 1/9 but stops stepping
  (fwd -6 cm) and leans forward 3.8 deg (b7 0.6): not a candidate. With the series give the mild penalty is still too
  strong for stepping (as h1); with the round-5 give it isn't (h2).
- h4_t05 (series give, tilt -0.05, eval_h4.txt): fitted 0/9, 32 cm, reach 10 / 15 mm, |roll| p95 5.9, cmd 4-8 Hz 2.3%
  (lowest); hard 8/9 sideways; old 0/9 35 cm. Close to h2_old; the series give, lightly penalized, walks.
- vx 0.02 (eval_vx02.txt): h2_old 0/9 30 cm, h4_t05 0/9 27 cm, fb2 2/9 (back), |roll| p95 5.2-5.5 vs 7.3.
- h5_h2hard (h2 + knee lean to -0.13, give x0.4-1.5; eval_h5.txt): fitted 1/9, 15 cm, reach 10 mm; HARD 3/9 (h2 7/9,
  h4 8/9, fb2 8/9), |roll| p95 5.3-5.4 in both; old 0/9 21 cm. The most robust stepping policy: half h2's progress.
- h6_h2s1 (h2's recipe, seed 1; eval_h6.txt): fitted 0/9, 23 cm, |roll| p95 5.8; hard 6/9; old 0/9. The recipe repeats
  (h2: 0/9, 32 cm, hard 7/9).
- 08:30 queue 8: h7_h5step (h5 + forward pay), the last run.
- h7_h5step (eval_h7.txt): fitted 0/9 11 cm, hard 2/9, leans forward 3.9: more robust but steps less than h2. Not better
  than h5.

## Morning summary (09:20, ~7.5 h GPU)
Robot candidates (full body, `--vx 0.015`), in order:
1. h2_old: fb2's stepping on the right masses, calmer sideways (fitted 0/9, 32 cm, |roll| p95 5.5 vs fb2 7.4; 0.02 too).
2. h5_h2hard: the most robust (hard sim 3/9 vs 7-8/9 for h2 / h4 / fb2), half the progress (15 cm).
3. h4_t05: the series give in training, the smoothest commands (4-8 Hz 2.3%), otherwise like h2.
Not for the robot: g1-g6, h1, h3, l1 (stand in place, or fall sideways). Legs: b7_seed2 stays the leg policy.
