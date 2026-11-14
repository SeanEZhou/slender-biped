# Round 7 (2026-11-13 afternoon, user away ~12:10-19:00): the ringing simulator

## Robot data (morning, logs in hardware/leg_p0/software)
- E8 wiggles (exercises.py wiggle): roll / pitch / yaw 1.5 deg, roll 3 deg, squat 1 deg lean 2, knees 1 deg lean 2.
  Pitch: 21-24 deg/s per deg of hip-pitch command at 4-5 Hz (sim 2-3); hip pitch follows 0.5 at 4.5 Hz, ~80 ms late.
  Roll / yaw servos acceleration-limited ~400-500 deg/s^2. Knees under load don't move for +-1 deg commands (sit 2.3 deg
  behind). Robot stands +4.6..+7 deg back at the nominal stance (back tip limit 7.5).
- Policy runs 6-10 as instructed (user: flags exactly as given): b7_seed2 x3 (11:52-11:53, low shake, falls left 4-8 s),
  h4 --gyro-scale 0.5 (11:53:33, :52), --gyro-scale 0.75 (11:54:17, 11:54:41 = 42 s), --act-lpf 0.6 (11:55:42, 11:56:00),
  h2 --gyro-lpf 2.5 --act-lpf 0.6 (11:56:58, 11:57:12, 11:57:34). User: gyro-scale 0.75 and act-lpf 0.6 "almost
  look like they could work reliably" but twitch violently when standing still.

## Simulator changes (code)
- evaluation.py / mjx_env.py: `ring` frame mode (damped oscillator per axis driven by the servo targets, torque on the
  pelvis), `ring_delay`, `dr_ring`; `gyro_lpf` observation option; `flex_dscale` (unused in the end); joint slices
  bounded to nj (qpos[7:7+nj]) so models with extra passive joints work.
- Fit (scripts/robot7_analysis/fitsim.py vs E8): pitch ring 4.2 Hz, zeta 0.2, gain 1.0, inertia 0.02; roll 5.3 / 0.25 / 0.7;
  command dead time 3 steps (60 ms) + ring delay 20 ms match the pitch response's size and phase (score 0.38 vs 1.08).
  Tried and dropped: softer hip-pitch give (no resonance, sags back), torso on hinges (models/hw_ank3_b150_m2uf_10dof:
  the pelvis IMU doesn't see it), less servo damping on the give (no resonance).
- Training stance: knee offset [-0.05, 0] (the robot stands +5 back; -0.09 put the sim past its tip point).
- Closed loop (eval6 --sim r7, 6 x 30 s): h2_old 3/6 falls, shake 14.8 (robot 20-30); h4 0/6, 10.1; b7 6/6 in 3 s
  (robot: b7 falls 4-8 s). Milder than the robot but the loop now matters.

## Robot runner (policy_stand.py)
- --gyro-lpf defaults to the run's gyro_lpf; history inputs (obs_hist); --ready-act (blend to the pose + action the
  policy holds standing, no snap at engage); --gyro-scale.

## Training queue
- queue0: c0_hist (b7 -> full body, 5-frame history, round-6 sim) 12:14-~12:44
- queue1: r1_raw, r2_lpf, r3_hist (ringing sim) ~12:45-14:20
- next: evaluate on --sim r7 / r7hard; continue the best; smoothness variant; seeds.

## Log
- 12:30 queue2 (r4_smooth, r5_histlpf, r6_hard) waits for queue1 (pid 1362494). eval6 --sim r7hard reproduces the robot: h4/h2/fb2 shake 18-24, 5/6 falls in 7-9 s.
- Runs 6-10 (lifts > 5 mm counted as steps): b7 full body 3.9-8.0 s, 1-5 steps, falls left; h4 gyro x0.5 8 s, 7-8
  steps; h4 gyro x0.75 16.5 / 42.3 s, 25 / 63 steps; h4 act-lpf 0.6 10.3 / 42.4 s, 17 / 65 steps; h2 gyro-lpf 2.5 +
  act-lpf 0.6 4.7-13 s, 4-15 steps. Morning filtered h4: 23-30 s, 35-47 steps. 4-8 Hz gyro still 12-24 roll. Falls
  mostly to the left (roll +22..+25).
- 12:35 watchdog: c0_hist at 10.2M (reward 1177 -> 6467, len 191 -> 669), GPU 97%; queue1/queue2 waiting.
- 12:55 c0_hist: r7 1/6 falls, 11 cm, shake 13.0; r7hard 5/6 (back), shake 26.6. r1_raw training (ring env compiles). robot7_autoeval.sh evaluates each r-run (r7 + r7hard, 9 x 30 s) into results/robot7/eval_<run>.txt
- 13:00 runner history layout checked against the evaluator (c0_hist, 150 steps): identical observations.
- 12:55 watchdog: r1_raw training 9 min in, GPU 98%, RAM ok.
- 13:17 r1_raw learns slowly (len 238->295, ~always falls). Queues 1-2 replaced by queue3: r3_hist, r1b_raw (+40M), r7_h4 (h4_t05 -> ring), r5_histlpf. Autoeval restarted for those.
- 13:35 r1_raw done (30M, len 337): r7 5/9 falls (4 left), 17 cm, shake 12.4; r7hard 7/9. r3_hist at 10M: len 540 -> 644, reward 3916 -> 5240 (better start).
- 13:58 r3_hist done: r7 4/9 (3 left), 13 cm, cmd4-8 4.6%; r7hard 6/9. Still chattering. queue4 after queue3: r8_alpf_h4 (act_lpf 0.6), r9_alpf_hist, r10_lcp_h4.
- 14:15 r1b_raw at 20M: len 333 -> 440, reward 3266 (still learning). GPU 98%.
- 14:35 r1b_raw (r1 +40M, train len 654): r7 4/9 falls, 4 cm, steps 2-5 mm, cmd4-8 5.0%; r7hard 5/9. Barely steps. r7_h4 started (len 531 at 0M).
- 14:58 r7_h4 done (len 531 -> 661; h4 itself reached 753 in the round-6 sim). r5_histlpf started. Waiting for r7_h4 eval.
- 15:25 r7_h4: r7 2/9 falls, 10 cm, cmd4-8 2.5%; r7hard 6/9 (back) -- no better than h4 itself (0/6, 5/6). Diagnosis (h4 on r7hard, 6 x 30 s): no ring 2/6; pitch ring alone 5/6 back; delay 2 instead of 3: 3/6; knee -0.025: 4/6. The pitch ring + 60 ms delay is the killer; RL fine-tuning (30-70M) doesn't learn a lower loop gain on its own. queue4 (act_lpf 0.6, LCP) cuts that gain structurally.
- 15:35 r5_histlpf: r7 3/9, 3 cm; r7hard 7/9 (back). r8_alpf_h4 started: len 598 at 0M (h4 in the ring sim without the filter: 531).
- 16:00 r8_alpf_h4 at 20M: len 742 (best so far in training). Baseline h4 + run-time act_lpf 0.6 (untrained, like the robot test): r7 3/9, 28 cm, cmd4-8 0.9%; r7hard 7/9.
- 16:17 r8_alpf_h4: r7 0/9 falls, shake 7.6, cmd4-8 0.9% -- steadiest -- but 3 cm (steps 0-4 mm); r7hard 3/9 (back). queue5 after queue4: r11_alpf_fwd (r8 + 2x forward pay), r12_alpf5_h4 (act_lpf 0.5), r13_alpf_b7 (b7 + act_lpf 0.6). r9_alpf_hist training (len 726 at 10M).
- 16:35 r9_alpf_hist: r7 2/9, 7 cm, shake 8.9; r7hard 5/9. r10_lcp_h4 training (LCP runs fine).
- 16:55 r10_lcp_h4 at 20M, len 643. GPU 98%.
- 17:15 r10_lcp_h4: r7 2/9 (left), 20 cm, cmd4-8 1.7%; r7hard 5/9 but 15 cm forward: LCP keeps the steps with less chatter. r11_alpf_fwd at 10M (len 750).
- 17:45 r11_alpf_fwd: r7 0/9, shake 6.4, cmd4-8 1.1%; r7hard 2/9 (back), shake 12.2 -- steadiest -- but 5.6 cm in 30 s (steps 0-4.5 mm); 2x forward pay didn't bring the steps back. r12_alpf5_h4 training.
- 17:55 r12_alpf5_h4 at 20M, len 738. r13 next (ends ~18:50).
- 18:20 r12_alpf5_h4: r7 0/9, 1 cm; r7hard 2/9. Lifts > 5 mm in 30 s on r7 (2 walks): r11 2 / 9, r12 6 / 4, r10_lcp 19 / (fell at 1 s), h4 18 / 17. The action-filter runs learned to mostly stand (steady, no shake, few steps). r13_alpf_b7 training (len 242 -> 592 at 10M: the filter makes the ringing sim learnable from the legs policy, r1 without it reached 337 at 30M).

## Summary (18:55)
- r13_alpf_b7: r7 0/9 falls, 17 cm, shake 6.9, cmd4-8 0.7%, lifts 8-16 / 30 s; r7hard 4/9. Best balance.
- Results table, learnings and the robot list: docs/walking_clean.md §21.
- Robot candidates: 1. r13_alpf_b7, 2. r10_lcp_h4, 3. r11_alpf_fwd (diagnostic: mostly stands). Optional --ready-act.
- Nothing committed. New checkpoints in runs/c0_hist, r1_raw, r1b_raw, r3_hist, r5_histlpf, r7_h4, r8..r13.
- 19:20 robot round 7 (user): r13 x4, r10 x5, r11 x1: shake tamed (7-10 / 4-9 deg/s) but all walk 7-12 deg forward and fall on their faces; h4 --act-lpf 0.6: 188 s (299 steps), 74 s, 76 s: best full-body walker; h4 --gyro-scale 0.75: 3.1, 3.8 s. walking_clean §22.
