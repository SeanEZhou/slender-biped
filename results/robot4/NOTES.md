# Robot round 4 notes (overnight 2026-11-12)

## Sim calibration on the E1 tilt test (scripts/tilt_test_sim.py; tilt_sweep1/2.txt)
Hand-like tilt (PD force at the pelvis, 1 deg/s), pelvis model, 5 V, robot-like servos:
- rigid: tip 13.8, slip 4.5, let go at 6 -> gap 81 -> 76 mm
- hip_roll give k=5: tip 9.1, slip 5.3, gap -> 52; k=4: 8.5 / 5.7 / 51; k=3: 5.6 / 6.5 / 51; k=2 falls by itself
- ankle_roll give adds little; friction 0.6 vs 1.0 barely matters
- robot: tip 6-7, slip 4-5, gap 85 -> 60-70. Chosen k = 4 (train randomized 3-6).

## Walking screen at k=4 (scripts/robot4_eval.py, 6 x 30 s, 5 V)
c3m 0/6 at 0.015 and 0.02, 2/6 at 0.03; c3f 0/6; h13 6/6; h9 6/6 -> matches the robot (c3m, c3f walk; h fall).
k=5 is too kind (h13 0/6). Gate = k=4.

## Outer pads (scripts/make_outpad_model.py; outpad_tilt.txt, outpad_walk.txt)
Tilt test with k=4 give, sideways tip angle:
- base 8.5; half-disc R30 on the outer edge 10.0 (R45 unstable: the disc's apex is a single point, the robot pivots
  on it and falls fore/aft); board shifted out 15 mm 10.3; outer strip (board widened outward, full length) 15 mm
  12.1, 25 mm 14.5; shift 10 + strip 15 12.5.
- Walking, k=4, vx 0.015: h13 6/6 -> 0/6 with a 15 mm strip; h9 6/6 -> 0/6. c3m at 0.03: 2/6 -> 0/6.
=> a full-length outer strip (straight outer edge) is the shape; a half circle is worse than a strip.

## Baseline c3m (k=4, 12 x 30 s; baseline_c3m.txt)
falls 1/12 @0.015, 2/12 @0.02, 5/12 @0.03, 9/12 @0.04; step ~20 mm at every speed; |roll|>4 14-21%.

## r1_flex (c3m + give, 20 M; eval_r1.txt)
falls 0/12 @0.015, 0/12 @0.02, 1/12 @0.03, 11/12 @0.04 (it was never commanded past 0.03); step 19-22 mm;
|roll|>4 9-13%. Steadier than c3m up to 0.03, same step length.

## r2_lim (r1 + roll_lim -1 per deg^2 past 4 deg + narrow -1 below 70 mm; eval_r2.txt)
0/12 falls at every speed, but it stopped walking: forward 0-13 cm in 30 s, steps 0-9 mm, gap widens to 103-110 mm.
The 4 deg limit at -1 is too strong (c3m spends 15% of its time past 4 deg). r3/r4 with the same terms were stopped;
queue 3 (scripts/robot4_queue3.sh) runs r3_long (no limit terms), r5_mild (-0.1 past 5 deg), p1_w15, r4_t15, p2_w25.

## r3_long (no limit terms, cmd 0.02-0.04; stopped at ~22 M -> runs/zz_r3_long_inplace)
20 M: 0/12 falls but forward ~0: it steps in place (training metrics: touchdowns 12.5 like r1, vx 0; stride reward
falling). r2's stall is the same thing, not (only) the roll limit: with the give, walking forward costs falls and the
stride term (20) doesn't pay for it. r1 trends that way too (training vx 10 -> 8.4).
Queue 4 (scripts/robot4_queue4.sh): progress=1 (box-averaged speed / command) + stride=40, from r1_flex:
s1_fwd, s2_long, s3_w15, s4_mild, s5_w25.

## s1_fwd (r1 + progress=1 + stride=40, 0.01-0.03, 20 M; eval_s1.txt) -- best so far
k=4: falls 0/12 @0.015, 0/12 @0.02, 0/12 @0.03, 2/12 @0.04 (c3m 1, 2, 5, 9); forward 54-62 cm (c3m 29-49);
step 21-26 mm (c3m 19-21); |roll|>4 15-26%; gap min 52-56 mm (narrows more than c3m's 53-64).
k=3 (softer than the robot?): s1 1/12 and 0/12; c3m 10/12 and 12/12; r1 1/12 and 2/12.

## s2_long (s1 recipe, commands 0.02-0.04, stride to 5 cm, 30 M; eval_s2.txt)
k=4: 0/12 falls at 0.015, 0.02, 0.03 AND 0.04; forward 61-70 cm; step 24-28 mm; |roll|>4 10-19%; gap min 66-68 mm
(keeps its stance better than s1 and c3m). k=3: 4/12 @0.015, 1/12 @0.02, 0/12 @0.03, 3/12 @0.04 (s1: 1, -, 0, -).
Best at the calibrated give; less margin than s1 if the real hip is softer than k=4.

## s3_w15 (s2 recipe trained ON the 15 mm outer strip; eval_s3.txt)
Worse than s2: k=4 0/12 @0.015-0.02 but 4/12 @0.03-0.04; k=3 3-5/12. It spends the strip's margin: its board gap
narrows from 96 to 64-66 mm (trained with narrow_min unset).
But s2_long (trained on plain boards) RUN on the 15 mm strip: 0/12 at k=4 AND k=3, 0.015 and 0.03, gap 74-84 mm.
=> best combination so far: s2_long + 15 mm outer strips (the strip as a safety margin, not something to train into).
Strips without retraining (outpad_walk.txt): c3m on the 15 mm strip 0/12 at k=4 and k=3 (plain boards at k=3:
10/12 and 12/12). s2 on the 25 mm strip: 0/12 at k=4 and 3, 22-27 mm steps. The strip alone fixes c3m's sideways
falls in the sim, and s2 adds ~5 mm a step and ~25% more distance.

## s4_mild (s2 + roll_lim -0.1 past 5 deg; eval_s4.txt)
Worse than s2: a fall at 1 s in one seed at every speed, slower (40-45 cm), 2-3/12 at k=3. The roll penalty, even
mild, costs more than it gives. (The negative mean step is the 1 s fall's episode skewing the per-episode average.)

## s5_w25 (s2 recipe trained ON the 25 mm strip; eval_s5.txt)
0/12 falls at k=4 and k=3, but short steps (16-19 mm), slow (35-43 cm) and it leans on the strip (|roll|>4 21-41%),
board gap narrowed 106 -> 70-80 mm. Training on the strip again spends its margin; s2 run on the strip is better.

## s6_wide (s2 fine-tuned 30 M, give k 2.4-6, commands 0.015-0.04; eval_s6.txt) -- new best
k=4: 0/12 at all four speeds; forward 69-75 cm (c3m 29-49); step 27-30 mm (c3m 19-21); gap min 65-71 mm.
k=3: 1/12, 1/12, 0/12, 1/12 (s2: 4, 1, 0, 3; c3m 10-12/12). |roll|>4 18-26% at k=4 (more sway than s2's 10-19).
s6 on the 15 mm strip: 0/12 at k=4 and k=3 (0.015, 0.03), steps 25-27 mm.

## s7_narrow (s2 fine-tuned 30 M, give k 2.4-6, commands 0.015-0.04, narrow -1 below 65 mm; eval_s7.txt) -- BEST
k=4: 0/12 at 0.015 / 0.02 / 0.03 / 0.04; step 20 / 20 / 25 / 33 mm (grows with the command, unlike c3m's flat 20);
forward 50 / 51 / 64 / 82 cm (c3m 49 / 47 / 41 / 29 with 1 / 2 / 5 / 9 falls); |roll|>4 10-20%; gap min 65-70 mm.
k=3: 0/12 at all four speeds (c3m 10/12, 12/12). k=2.5 (much softer than measured): s7 3-4/12, s6 2-4/12, c3m 12/12.
s7 on the 15 mm strip: 0/12 at k=4, 3 AND 2.5 (0.015, 0.03).

## s8_push (s7 + progress 2, stride 60, 30 M; eval_s8.txt)
k=4: 0/12 at all speeds; steps 27-31 mm; forward 69-81 cm. k=3: 3-4/12 (s7: 0). On the 15 mm strip: 0/12 at k=4 and
3, steps 26-32 mm. Longer steps than s7 at slow speeds, less margin on plain boards: s8 is the choice WITH the strips,
s7 without.

## Pitch-offset check (the robot stands 3.7 deg back; IMU pitch bias +-0.065 rad, k=4, robot4_eval --set)
s7: 0/12 either sign at 0.015 and 0.03. c3m: 2/12, 4/12 (+) and 5/12, 6/12 (-). s7 tolerates the offset; no
pitch trim needed.

## s9_t15 (from s7 on the slower c_T15 gait, 1.5 s a step, 30 M; eval_s9.txt)
The calmest: |roll|>4 only 6-12% (s7 10-20%); 0/12 at k=4 all speeds, 0/12 at k=3 up to 0.03 (1/12 at 0.035).
Slower and shorter steps at low commands (13-18 mm at 0.015-0.02, 26 mm at 0.035; c_T15 tops out at 0.035).
s9 at k=2.5: 3/12, 2/12 (s7 3/12, 4/12).

## Strip fit check
F_outer_strip15/25 against A02_foot_ankle (flattened): zero overlap volume on either edge (manifold boolean); the
lip clears the servo socket and rib. Render: hardware/feet_p1/renders/outer_strip15_on_foot.png.

## s10_robust (s7 + give k 2-6 + IMU tilt dr 0.05, 30 M; eval_s10.txt)
k=4: 0/12 at all speeds, steps 26 / 27 / 29 / 32 mm (s7: 20 / 20 / 25 / 33), forward 64-80 cm. k=3: 0/0/0/2.
k=2.5: 4/4/4/7 (s7 3-4 at 0.015/0.03). Longer steps at slow speeds than s7, same margin down to k=3; not more
margin below it. Randomizing the give below k=3 didn't buy robustness there.
s10 on the 15 mm strip: 0/12 at k=4, 3, 2.5. With the 3.7 deg pitch offset: 0/12, steps 30-32 mm.

## s11_pushn (s8 + narrow x3 + give k 2-6; eval_s11.txt)
0/12 at k=4 and k=3 at all speeds, but the steps shrank back to 20-25 mm (s8: 27-31). k=2.5: 4-7/12.

## Summary (07:55 -> written 08:20)
Recommended for the robot: s10_robust (0 falls at k=4 and k=3 up to 0.03, steps 26-32 mm, tolerates the 3.7 deg
pitch offset), then s7_narrow (calmer at slow speeds, 20-33 mm). With 15 mm outer strips: s8_push (27-31 mm steps,
0/12 at k=4/3) or s10. c3m for reference: 1-9/12 falls at k=4, 10-12/12 at k=3, 20 mm steps.
