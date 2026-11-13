# Robot round 5 (2026-11-12, user away ~7 h)

Goals: (1) full-body policies from the round-4 recipe, steadier than c3f_T12 and stepping forward;
(2) legs: fit the sim to today's robot walks (walking_clean §16: falls backward after 7-28 s, walks 1-3 deg back
creeping to +4..+6, L reach > R), then train on it.

Robot reference (scripts/gap_fit.py --logs, 4 trials each at 0.015):
  s7 T 17.6 s, s8 14.8, s10 23.6, c3m 13.6; pitch early +1.2..+2.4, late +2.6..+5.2; reach L 12-19 R 6-15 mm

GPU queue 1 (scripts/robot5_queue1.sh): fb1_s10, fb2_s7, fb3_c3f on hw_ank3_b150_10dof, 30 M each.

## Fitting the simulator to the walks (CPU, scripts/gap_fit.py; results/robot5/fit*.txt)
- Servo tracking on the robot is fine (hip pitch / knee within 1-2 deg of command, 80-140 ms lag; knees sag +1.5 deg).
- Falls: at each landing the body pitches back (to -96 deg/s, +8..+14 deg) and doesn't come back.
- Rigid-lean sim (k=4 only): no falls, pitch -1..-2 (forward). Robot: falls 14-24 s, pitch +1..+5 (back).
- Hidden knee offset -0.045 rad: stance +3.7 deg back as the robot; walking pitch +3..+4; pitch wobble matches (7-8 deg p5-95).
  Alone: few falls.
- Backward pull (cable / rug), F = min(k * distance, cap): 2 N/m cap 0.6 all fall in 10-16 s (too strong).
- **Fit: knee -0.045 + 0.6 N/m capped 0.2 N**: falls s7 15.7 s, s8 18.3, s10 22.8, c3m 20.2 (robot 17.6 / 14.8 / 23.6 / 13.6),
  mostly backward, late pitch +4.5..+5.3. Misses: c3m too good (robot: sideways falls), steps still ~2x the robot's.
- c3f on the full body: k=4 0/12 falls, 44 cm / 30 s; fitted (knee + pull): 6/8 backward, 16 s, 12 cm.

GPU queue 2 (scripts/robot5_queue2.sh, after fb1_s10): b1_lean (s10 + lean + pull), fb2_lean (fb1 + lean + pull),
b2_lean_s8, fb3_c3f (c3f + give + lean + pull).
- Hip give k=3 (with knee + pull): c3m falls sideways in 4.8 s (7/8 left), s10 23.7 s. **k=3.5 (n=12): s10 24.8 s, s7 21.1,
  s8 19.8, c3m 17.2 (back 5, left 4)**: the robot's order (23.6 / 17.6 / 14.8 / 13.6) and c3m's mixed falls.
  => FITTED sim = hip_roll k 3.5 + knee -0.045 + pull 0.6 N/m cap 0.2 N. Yardstick for new policies:
  gap_fit.py <runs> --n 12 --set 'hidden_off=[["knee",-0.045]]' drag_back=0.6 drag_back_max=0.2 'hidden_flex=[["hip_roll",3.5]]'
- Step length gap: joint ranges match (s10 hip pitch 23-25 deg, knee 23-25, robot and fitted sim alike), so the robot's
  shorter reach is timing / contact: encoder swing 0.45-0.5 s on the robot vs ~0.6 in sim. Likely early touchdown on the
  rug (less real clearance: E4). For tonight's measurements.

## fb1_s10 (full body from s10, s10 recipe, no lean/pull) - results/robot5/eval_fb1.txt
- k=4: 0/12 at 0.015 and 0.03; 55-60 cm / 30 s, 20 mm steps (c3f: 0/12, 44-53 cm, 14-19 mm). Sways more: |roll|>4 deg
  27-34% (c3f 18%), gap min 61 (c3f 76).
- k=3: 2/12, 1/12 (c3f 3/12, 2/12).
- Fitted sim (k 3.5 + knee + pull): fb1 8/12 back, 20.3 s, 30 cm; c3f 7/12, 20.4 s, 22 cm.
- => steps forward better than c3f, not steadier yet. fb2_lean (from fb1, lean + pull) is the one meant to be.

## b1_lean (s10 + lean + pull, legs) - results/robot5/eval_b1.txt
- Fitted sim (k 3.5 + knee -0.045 + 0.6 N/m cap 0.2): **0/12, 30 s, 56 cm, pitch +0.2 -> +1.1** (s10: 5/12, 24.8 s, 36 cm, +3.3 -> +4.1).
- Harder (k 3, knee -0.07, 1.0 N/m cap 0.3): 4/12, 24 s, 39 cm (s10: 11/12, 15.6 s, 5 cm).
- Robot-like k=4 / k=3 (robot4_eval): 0/12 everywhere; steps 16-20 mm (s10 26-32); |roll|>4 10-23%; gap min 63-69.
- Plain sim: 0/12, pitch -3.2 (leans forward to offset the lean it expects).
- => the first candidate for the robot from this round. Cost: shorter steps than s10.

## fb2_lean (full body from fb1, lean + pull) - results/robot5/eval_fb2.txt
- Fitted sim: **1/12, 29.6 s, 48.5 cm, pitch -0.8 -> +0.4** (c3f 7/12, 20.4 s, 22 cm; fb1 8/12, 20.3 s, 30 cm).
- Robot-like k=4: 0/12 (0.015), 1/12 (0.03); k=3: 1/12, 0/12. 37-47 cm, steps 15-19 mm, |roll|>4 12-19% (c3f 18-24%), gap min 61-68.
- Plain sim: 1/12, 48.6 cm (c3f 1/12, 44.1 cm).
- => the full-body candidate: steadier than c3f in the fitted sim, twice its forward progress there. (c3f's sim progress
  didn't show on the robot; the fitted sim's 22 cm for c3f is closer to its marching in place.)

## b2_lean_s8 (s8 + lean + pull, legs) - results/robot5/eval_b2.txt
- Fitted sim: 0/12, **67 cm**, reach 23-25 mm (b1_lean 0/12 56 cm, s8 8/12 31 cm).
- Harder (k3, knee -0.07, 1.0 cap 0.3): 8/12, 19.5 s (b1_lean 4/12).
- Robot-like k=4: 1/12 at each speed, both at 2 s (a first-step fall); "step" 5-7 mm = many extra touchdowns (foot
  chatter?) though 40-47 cm; k=3: 0/12, 18-22 mm.
- => longer steps, less margin than b1_lean. b1_lean stays first for the robot; b2 if b1 works and steps are wanted.

## fb3_c3f (full body from c3f, give + lean + pull) - results/robot5/eval_fb3.txt
- Fitted: 2/12, 38 cm (fb2_lean 1/12, 48 cm). Harder: 10/12, 5 cm (fb2_lean 6/12, 25 cm).
- Robot-like k=4 0/12 (44 cm, 16 mm), k=3 1/12 each (3 s), |roll|>4 19-27%.
- => fb2_lean stays the full-body pick. Queue 3 running: b3_lean_long, b4_lean_soft, fb4_soft (from fb2_lean).

## b3_lean_long (b1_lean + progress 2, stride 60) - results/robot5/eval_b3.txt
- Robot-like (no lean): 0/12 everywhere, **84 cm, 33 mm steps** - looks like the best walker yet.
- Fitted sim: **8/12**, 19.7 s (b1_lean 0/12). Harder: 11/12.
- => the forward pay buys steps by leaning on margin the lean and pull take away. Not for the robot. The fitted sim is
  what separates it from b1; the old screen would have recommended it.

## b4_lean_soft (b1_lean + softer give, stronger lean / pull) - results/robot5/eval_b4.txt
- Robot-like (no lean): 0/12 everywhere, 64-71 cm, 26-28 mm steps.
- Fitted: **6/12**, 21 s (b1_lean 0/12). Harder: 12/12 at 9 s. Worse despite training on harsher conditions.
- b1_lean on 24 seeds, fitted: 1/24, 50 cm; fb2_lean (full body) 2/24, 43 cm: both hold up.
- => training outcome varies run to run (the c3m lesson); queue 4 checks whether b1 / fb2 reproduce with another seed.

## fb4_soft (full body, fb2_lean + softer give, stronger lean / pull) - results/robot5/eval_fb4.txt
- Fitted (24 seeds): 4/24, 47 cm (fb2_lean 2/24, 43 cm). Harder: 6/12 (fb2_lean 6/12).
- Robot-like: 0/12 at k=4, 1/12 at k=3 0.03 (at 29 s); 57-60 cm, 20-22 mm steps (fb2: 37-47 cm, 15-19 mm); |roll|>4 14-22%.
- => full body: fb2_lean the steadier, fb4_soft the longer steps at about the same margin.

## b5_seed1 (b1_lean's recipe and start, seed 1) - results/robot5/eval_b5.txt
- Fitted (24 seeds): **10/24**, 49 cm (b1_lean 1/24). Harder: 12/12. Robot-like k=4: 0/12, 72-83 cm, 28-32 mm steps.
- => b1_lean doesn't reproduce: the same recipe gave a longer-stepping, less steady policy (like b3, b4). Training
  outcome is luck-dependent (the c3m lesson); the fitted sim is what picks the lucky one before the robot does.
  Queue 5: one more seed (b7_seed2) to see how often the recipe lands.

## fb5_seed1 (fb2_lean's recipe and start, seed 1) - results/robot5/eval_fb5.txt
- Fitted (24): **12/24**, 49 cm (fb2_lean 2/24). Harder 12/12. Robot-like k=4: 0/12, 67-79 cm, 24-27 mm steps, |roll|>4 22-27%.
- => same as b5: seed 1 lands on longer steps and little margin; seed 0 (b1, fb2) on shorter steadier steps. Every
  policy that steps 25+ mm in the old screen this round (b3, b4, b5, fb5) fails the fitted sim.

## b6_mid (b1_lean + progress 1.5, stride 50) - results/robot5/eval_b6.txt
- Fitted (24): **0/24**, 52 cm. Harder: **2/12**, 45 cm (b1_lean 4/12). Robot-like k=4: 0/12, 47-49 cm, 19-20 mm, |roll|>4 7-11%.
- => the leg pick (b1_lean the alternative). Steps no longer than b1, but steadier.

## b7_seed2 (b1_lean's recipe and start, seed 2) - results/robot5/eval_b7.txt
- Fitted (24): **1/24, 60 cm**. Harder: 6/12. Robot-like k=4: 0/12, 61-72 cm, **27-29 mm steps**, |roll|>4 22%.
- => passes, with s10-length steps: the longer-step candidate. The recipe passes on 2 of 3 seeds (0, 2; seed 1 fails).
- Queue 6: b8_mid2 = b7 + b6's milder forward pay (as b6 did for b1).

## b8_mid2 (b7_seed2 + progress 1.5, stride 50) - results/robot5/eval_b8.txt
- Fitted (24): 3/24, 67 cm. Harder: 9/12. Robot-like k=4: 0/12, 64-69 cm, 26-28 mm, |roll|>4 9-11%.
- => less margin than b7_seed2; b7 stays the longer-step pick.

## Picks for the robot
- Legs: b6_mid (0/24 fitted, 2/12 harder, 19-20 mm), b1_lean alternative; b7_seed2 for s10-length steps (1/24, 6/12).
- Full body: fb2_lean (2/24, 6/12, 15-17 mm, calm); fb4_soft for longer steps (4/24, 6/12, 20-21 mm).
