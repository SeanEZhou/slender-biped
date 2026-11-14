# Round 8 (2026-11-13 night, 1 h of GPU): why the round-7 policies lean forward; h4 with the filter trained in

## The lean, robot vs simulator (scripts/robot8_analysis/lean.py, stance.py, stance_tau.py, gains.py)
- Robot logs (evening): before engage every policy stands at +4.1..+7.5 deg, knee-hip 17.2, pitch - (knee-hip) = -12.
  The simulator's stance gives that at knee offset ~-0.04 (r7 fit -0.025: -13.2): statically the two agree.
- Walking, pitch - (knee-hip) (a foot proxy, flat board = -12..-14): sim -12..-15 for every policy on every screen;
  robot h4 -13..-17, r13 -16..-21, r10 -18..-22. The round-7 policies roll onto their toes on the robot only. Their
  knee-hip also drops (r10 7-10 against sim 9.8).
- Servo load walking: r10 hip pitch +54/-24, knees -19/+15 (unloaded); h4 +6/-2.5, knees -105/+68 (as standing).
- Sim conditions tried on h4 / r10 / r13 (3 x 15 s each): knee offset 0 / -0.04 / -0.09, soft rug, weak servos, hard
  ring, no ring + short delay, low friction, mass x1.2: none tips them past -5 to -7.6 deg.
  What does: the pelvis centre of mass forward (1 cm: -5.3 / -6.8 / -5.8; 2 cm: -9.2 / -8.2 / -7.7, knee-hip 8-9,
  foot proxy -16..-18.5; 3 cm: falls forward) or a 0.3 N forward push (same picture). Pitch 0.5 s after engage:
  robot -6..-9 for all; sim r7 -3.5..+0.3, sim com +2 cm -2.6..-6.4, +3 cm -4.9..-8.2.
- The model stands at the robot's standing pitch (+5..+7, knee -0.04, zero action) in 1/6 draws at com 0, 3/6 at
  +1 cm, 6/6 at +2 cm; the robot stands there for seconds (tip limit 7.5 deg). => the real mass sits ~1-2 cm further
  forward than the model's: a forward bias that is in no training run.
- Gains on h4's 188 s robot walk (gains.py; knee-hip answer per deg of back tilt / pitch-rate rms gain hip, knee):
  h4 -0.57 / 0.047, 0.096; r10 (LCP) -0.14 / 0.017, 0.033; r13 -0.66 / 0.052, 0.071; r8 (h4 + filter trained in)
  -0.71 / 0.069, 0.114; r11 -0.94; h2_old -0.69. r10's Lipschitz penalty cut its posture stiffness to a quarter: it
  sags furthest. r8 raised its gyro gain by ~40% to make up for the filter (the filter takes 5 Hz gain away, the
  policy puts some back). r13 / r11 are as stiff as h4 and still lean: their gait, not their stiffness.

## Training (scripts/robot8_queue0.sh, robot8_queue1.sh)
- s1_alpf_com: h4_t05 -> h4's round-6 sim, act_lpf 0.6 in training, pelvis mass +[-0.5, 2.5] cm (dr_com_x, new), 25M
- s2_alpf_com_ring: s1 + round 7's ring and 2-3 steps of delay, 22M

## Baselines on the round-8 screen (results/robot8/screen_*.txt, 6 x 30 s)
- h4 + act_lpf 0.6 (the robot's best): r7 1/6, 37 cm, shake 8.3, cmd4-8 0.8%; r7+com1.5 1/6 (fwd) 25 cm;
  fitted+com1.5 0/6 20 cm; r7hard 5/6 (back), shake 16.5.
- h4 raw: r7 0/6, 31 cm, 10.1, 2.0%; r7+com1.5 3/6 (fwd); fitted+com1.5 0/6, 13 cm; r7hard 5/6, 19.7, 5.1%.

## s1_alpf_com (25M, train len 720 -> 847)
- r7 1/6, 30 cm, shake 10.2, cmd4-8 1.1%; r7+com1.5 3/6 (fwd); fitted+com1.5 0/6, 37 cm; r7hard 6/6 in 2.8 s (back).
- Gains: knee-hip per deg of tilt -1.08 (h4 -0.57), pitch-rate hip 0.090 knee 0.161 (h4 0.047 / 0.096). With the
  filter trained in and no ring in the simulator, the policy doubled its gains to undo the filter: worse than h4 +
  the run-time filter on every ringing screen. Not for the robot.

## h4 with other run-time filters (screen_h4_t05_lpf0.{5,7}.txt)
- act_lpf 0.7: r7 1/6, 33 cm, shake 7.9, cmd4-8 0.5%; r7+com 2/6; fitted+com 0/6 24 cm; r7hard 5/6, shake 13.9 (0.6: 16.5).
  A little calmer than 0.6 at the same falls: worth a robot trial (`--act-lpf 0.7`).
- act_lpf 0.5: r7 1/6, shake 9.6, cmd4-8 1.2%; r7hard 6/6: worse.

## s2_alpf_com_ring (stopped at 11M, 21:23; runs/s2_alpf_com_ring/params.pkl = the 11M checkpoint)
- r7 0/6, 17 cm, shake 7.4, cmd4-8 0.8%; r7+com1.5 1/6, 2.5 cm; fitted+com1.5 0/6, 5 cm; r7hard 2/6 (h4+filter 5/6), shake 14.0.
- Gains: knee-hip per deg of tilt -0.63, pitch-rate hip 0.060 knee 0.101: near h4's. The ring stops the filter's
  gain creep that s1 showed. But the steps fade as r8's did.
- Why they fade (progress.jsonl, eval episode terms, first -> last): r8 ref_swing 310 -> 115, stride 190 -> 35,
  progress 244 -> 73, ref_contact 1572 -> 1961. On the step-settle-step schedule both feet are down most of the
  time, so standing collects most of the contact pay; with the ring, stepping is the risky part.
- s3_swing: s2 11M + ref_swing 15 (6), ref_contact 2 (3), 24M, 21:24-~21:51.

## s3_swing at 12M
- lean.py (seeds 82000+): r7 1/6, 26 cm, 19 lifts / 30 s; r7+com1.5 2/6 (F, R), 16 cm, 16 lifts. (h4 + filter: 3/6, 12 cm,
  25 lifts; 2/6, 25 cm, 17. s2 11M: 1/6, 17 cm, 15; 2/6, 3 cm, 8.)
- screen (eval6 seeds): r7 0/6, 30 cm, shake 8.0, cmd4-8 0.9%; r7+com1.5 2/6 (fwd), 15 cm; fitted+com1.5 3/6 (fwd), 28 cm;
  r7hard 2/6, 12 cm forward, shake 12.4: the only one that walks forward on r7hard.
- Gains: knee-hip -0.76, pitch-rate hip 0.071 knee 0.116 (h4 0.047 / 0.096): creeping up again.

## Robot 22:16-22:30 (walking_clean §24)
- h4 --act-lpf 0.7: vx 0.015 147 s / 251 steps, 138 s / 227 (standing); vx 0.02 (by replay) 84 s, 67 s, 32 s (fell right in a bout).
- s3_swing: 38 s, fell left; in a bout 51% of the time. s2: 46 s, 59 steps in place, calm.
- The gyro gain on h4's robot log predicted the twitch ranking; the r7hard screen did not.
