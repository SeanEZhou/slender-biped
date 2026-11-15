# Round 12 (2026-11-14 22:35-23:35): the robot falls left; sideways mass range training

## Robot + video before training (k5, 16 runs 21:49-22:39; settings A guard on, B guard + `--gate-roll 6`, C `--no-gate-guard`)

Operator: the left foot steps toward the centre line instead of outside; the robot pivots right and already leans left
when the left foot lands; leaning further, the fall is inevitable.

- **Every fall since the video runs went to the robot's left** (7 of 8 new runs fell, roll ~-20°). A / B / C made no
  clear difference; the gate guard released a hold once in 7 runs, so these falls are not the frozen-clock kind.
- Roll at landing (encoder FK touchdown; - = robot's left): **left landings median -3.3° (p10 -7.1), right landings
  +2.0° (p90 +4.0)**; the left-landing lean grows within a run (about -2.5° on the first, -6 to -8° by the third / fourth).
- Encoder-FK step width at landing is the same for both feet (122 vs 124 mm, left a bit more variable, p10 110 vs
  117). The narrow left step the video shows (223822: the left swing crosses in front of the right leg, the feet end
  nearly in line) is therefore not in the joint angles: it is the body's lean and the hip's invisible give.
- IMU roll zero: standing roll at engage 0.0° median (p10 -0.5, p90 +0.6, 107 runs). Not an IMU offset.
- Hip roll under single support: the left hip roll's load is ~2x the right's (median 68 vs 36, servo units), its
  tracking error a little wider (p10 -1.96 vs -1.45°). Consistent with more weight on the left, or the left stance foot
  further out. **Hardware check worth doing: where the robot's centre of mass sits sideways (cables, board, wiring), and
  whether the left hip roll is looser.**
- Fitted sim, k5, pelvis mass moved sideways (6 x 30 s, vx 0.015, filter 0.7, +1.5 cm forward): none 2/6, 1 cm right
  0/6, 1 cm left 2/6 (both left), 2 cm left **4/6, all left**. k5 trained with only +-5 mm sideways.

New: `dr_com_y` (training, sideways pelvis mass range) and `com_dy` (evaluation). Runner gate guard (default on).

## Runs (from k5_high2 at 16M, round-10 recipe and fitted hips; `scripts/robot12_*.sh`)

Fitted sim, 12 x 30 s, vx 0.015, filter 0.7, mass +1.5 cm forward:

| run | change | training episode length | mass centred | mass 1.5 cm left |
|---|---|---|---|---|
| k5_high2 | | 468 at 16M | **2/12**, 35 cm | **6/12** (all left), 21 cm |
| q1_comy | sideways +-2 cm + fore-aft -0.5..+2.5 cm, 16M | 378 -> 358 | 5/12, 30 cm | 8/12 (7 left), 18 cm |
| q2_comy_wide | q1 + boards >= 10 cm apart, 14M | 391 -> 431 | 5/12, 40 cm | 7/12 (all left), 22 cm |
| q3_comy_only | sideways +-2 cm only, 12M | 429 -> 353 | 4/12, 38 cm | 9/12 (8 left), 25 cm |

**None beats k5.** A +-2 cm sideways mass range is more than k5's high step handles; 12-16M of fine-tuning in it made
the policy worse even with the mass centred (the same pattern as round 11's looser hips). Nothing new for the robot.

## Next

- Find the robot's sideways bias on the bench (centre of mass, left hip roll) and fix or measure it; then train on that
  offset (a narrow range around it) instead of a wide symmetric one.
- If it must be trained: a narrower range (+-1 cm) for longer, or reach it gradually.

## Hardware fixes and k5 again (2026-11-14 23:50 - 2026-11-15 00:55)

Hip-yaw servo screws tightened (all screws); the left hip pitch re-zeroed standing (`biped_bench.py trim`; net +1.5 deg,
offset 993 -> 969; knee -1974 -> -1949, unchanged). k5, `--vx 0.015 --act-lpf 0.7`, guard on / off similar:

| batch | trials | median up | median steps | falls | shake (deg/s) |
|---|---|---|---|---|---|
| before the screws (21:49-22:39) | 19 | 7.8 s | 4 | 14: 12 left, 2 back | 17.7 |
| screws (23:53-00:17) | 13 | 10.2 s | 9 | 11: all left | 14.3 |
| screws + re-zero (00:39-00:53) | 17 | 25.7 s (max 32.9) | 30 (max 42) | 7: 4 left, 3 right | 14.8 |

Landings after the re-zero: L roll -1.9 deg, width 132 mm (p90 153), reach +2 mm (p90 +17); R +0.9 deg, 126 mm (p90
140), +16 mm (p90 +25). Operator: some trials drift left with a much larger, outward left step. k5 is the new best.
