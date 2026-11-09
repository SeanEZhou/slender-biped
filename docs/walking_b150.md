# Walking on the F_board150w pads (2026-11-08)

The big snowboard pads ([foot_pads.md](foot_pads.md)) make standing on a rug nearly passive, so the next demonstration
is walking with real steps: the board lifts clear of the rug every step, not a shuffle. This round trained walkers for
the legs-only robot (as tested so far) and for the full body (torso and head on), plus a "safe" walker that trades speed
for consistency. It also tried a skiing policy, which did not work for a reason in the leg geometry (§4).

About 4.5 GPU-hours, 16 runs. Every number below is simulation on the CAD-derived models
(`hw_pad_board150w_legs_8dof`, `hw_pad_board150w_8dof`) with the fitted servo model, its delay and acceleration
randomization, and the rug floor model. On the robot these walkers fell on the first or second step; the
robot trials and everything tried since are in [walking.md](walking.md).

## 1. Results

Walk tests: 20 s at a fixed forward command with the heading held, the robot's IMU filter in the loop (±1000 °/s gyro,
accelerometer gated at 0.15 g), 10 seeds per rug, servo and mass randomization per seed. Rugs: soft contacts that sink
the original pad ~3 / ~9 / 9+ mm (firm / medium / deep; the big board sinks less).

**Legs-only robot**

| policy, command | rug | survives 20 s | forward speed | step | clearance | air time | body sway (RMS tilt) |
|---|---|---|---|---|---|---|---|
| walker `b150_walkstride_L`, 0.16 m/s | firm | 80 % | 0.028 m/s | 2.2 cm | 16 mm | 360 ms | 7.1° |
| | medium | **100 %** | **0.035 m/s** | 3.1 cm | 16 mm | 320 ms | 6.4° |
| | deep | 70 % | 0.029 m/s | 2.9 cm | 15 mm | 280 ms | 6.9° |
| safe walker `b150_walksafe_L`, 0.08 m/s | firm | **90 %** | 0.014 m/s | 1.7 cm | 16 mm | 360 ms | 5.6° |
| | medium | 90 % | 0.026 m/s | 2.3 cm | 15 mm | 325 ms | 4.9° |
| | deep | **90 %** | 0.027 m/s | 2.4 cm | 13 mm | 275 ms | 5.5° |

Both walk at 2.4 steps/s with the board clear of the rug for about a third of a second each step. They are slow,
1.4–3.5 cm a second, at best a board length (150 mm) every 5 s. The walker covers more ground; the safe walker
survives more consistently and sways less. On a hard floor neither is reliable (the walker survived 1 of 4 test walks
without the IMU filter): they were trained on rugs only.

**Full body (torso and head on, 1.26 kg, centre of mass 8 cm higher)**

| policy, command | rug | survives 20 s | forward speed | step | clearance | air time | body sway |
|---|---|---|---|---|---|---|---|
| safe walker `b150_walksafe_F2`, 0.08 m/s | firm | 70 % | 0.004 m/s | 1.5 cm | 14 mm | 335 ms | 6.4° |
| | medium | 70 % | 0.010 m/s | 1.5 cm | 12 mm | 300 ms | 6.0° |
| | deep | 80 % | 0.010 m/s | 1.3 cm | 10 mm | 260 ms | 5.5° |

With the torso and head the walker mostly steps in place, and 2–3 walks in 10 fall. The path there: the legs-only
walker fine-tuned on the full-body model (`b150_walk_F`, 20 M steps) survived only 17–50 % with the IMU filter (83 % on
the medium rug without it: the heavier, taller body is far more sensitive to the filter's lag than the legs alone), then
55 M steps of safe-walker training with the IMU filter (`b150_walksafe_F`, `b150_walksafe_F2`) took it to 70–80 %. Its
20 M checkpoint walks twice as fast (0.010–0.022 m/s) but falls more on the deep rug (50 %).

## 2. What it took

1. **A reference gait the boards can follow.** The round-4 reference (`slender_biped/ref_gait.py`) moves the foot under a
   pelvis held at constant height. Without ankles the board is rigid on the shin, so sweeping the foot back tilts the
   board ±5°, putting the robot on one edge of a 150 mm plank. On that reference the policy marched in place for 40 M
   steps (two runs). The new `--flat` mode keeps the stance board level and lets the pelvis bob instead (a level sole
   fixes the shin angle; the thigh swings about the knee, the pelvis rises and falls ~6 mm).
2. **Smooth it.** The level-sole IK switches task at each hand-over (single stance: x, y, level; double support: also the
   floor height), which made the knee jump 11° between two table samples, a knee acceleration of ~1400 rad/s² that the
   servo profile cannot follow. Smoothing the table around the cycle (circular Gaussian, 2 samples) brings it to ~160
   rad/s², lower than the old reference, for a slightly less level board at the hand-overs (3.8° worst).
3. **Make progress worth more than marching.** Once walking, the policy drifted back to stepping in place after 20–30 M
   steps (round 5 saw the same). The stride term only pays at touchdowns, so at weight 1 it was worth ~0.05 per control
   step against several points for tracking and imitation; weight 8 (a full 4 cm step ≈ 0.4 per control step) restored
   the forward steps.
4. **Command more than you want.** With the robot's IMU filter the walker reaches about a fifth of its speed command;
   0.16 m/s gives 0.03 m/s with 100 % survival on the medium rug.

## 3. What did not help

- **Training with the IMU filter** (`b150_walkimu_L`): it kept its balance but went back to stepping in place
  (−0.004 to 0.015 m/s). The same was true for standing (foot_pads.md §4). Evaluated with the filter, the walker trained
  without it walks better.
- **Fine-tuning the marcher with a sharper speed reward and the stride term** (`b150_walk_L2`): still marching.
- **Longer training**: steps shrink again after ~20 M steps; checkpoints are picked on the gait tests, not reward.

## 4. Skiing: why it does not work without ankles

Five runs tried a cross-country "classic" stride, both boards always on the floor and sliding alternately: twice from
scratch (the first on the old reference), from a standing skier, from the walker, and on a smooth low-friction floor
(μ 0.15–0.4) instead of a rug. All five settled on standing still with both boards down (0.000–0.003 m/s), whatever the friction.

The reason is kinematic. Each leg has two joints in the sagittal plane (hip pitch, knee) and the board is rigid on the
shin, so a board flat on the floor fixes the shin angle, and with the pelvis level that fixes how far forward the foot
sits for a given pelvis height. Two boards flat on the same floor are therefore always side by side; staggering them by
3 cm tilts at least one by ~5°. A skier's alternating glide is exactly that stagger, so every stride rocks the boards
onto their edges, and the policy finds standing still the better deal. Walking avoids it by lifting the moving board.
Skiing would need an ankle pitch joint. As agreed, the safe walker replaces the skier as the reliable fallback.

## 5. On the robot

`hardware/leg_p0/software/policy_stand.py run` now plays walking policies: `--vx` sets the forward command, the heading
is held by integrating the gyro's yaw rate (yaw-rate command = clip(−gain × heading change, ±0.3), as in training),
and the reference gait is looked up from `models/ref/<table>.npz` as the action base. Checked: the runner's lookup
matches the training code exactly.

```bash
python policy_stand.py run --run b150_walksafe_L --enable --ramp 1 --vx 0.08      # safe walker, legs only
python policy_stand.py run --run b150_walkstride_L --enable --ramp 1 --vx 0.16    # walker, legs only
python policy_stand.py run --run b150_walksafe_F2 --enable --ramp 1 --vx 0.08     # safe walker, torso and head on
```

Notes:
- Start on the rug, on the new pads, with the new IMU firmware.
- The runner's leg-collision rule (both hips may not swing inward together beyond ~5°) stays on; the simulator models
  the pad collision, so a walker should rarely hit it, but if a step looks clipped, that is why.
- `--ramp 1` blends the gait in over a second; the stance policy needs no ramp, but a walker starting at full stride
  from a standstill is the most likely moment to fall.

## 6. Files

- `slender_biped/ref_gait.py`: `--flat` level-sole reference with `--smooth`; tables `models/ref/b150L_flats.npz`
  (walking, legs only), `b150F_flats.npz` (full body), and the ski tables kept for the record (`b150*_ski*.npz`).
- `scripts/train.py`: tasks `walk_b150`, `ski_b150`; `scripts/b150_run.sh`: the training recipe.
- `slender_biped/mjx_env.py`: `airborne` reward term and `airborne_h`, `air_frac` / `swing_clear` metrics.
- `scripts/gait_quality.py`: `--floor-tc`, `--friction`, `--imu`, `--tag`; `scripts/render_gif.py`: `--floor-tc`,
  `--vx`, `--imu`, bundled ffmpeg fallback.
- Runs: `runs/b150_walkstride_L` (walker, `params_0020M`), `runs/b150_walksafe_L` (safe walker, `params_0015M`),
  `runs/b150_walksafe_F2` (full body, `params_0030M`). GIFs: `assets/gifs/b150_walk_L.gif`, `b150_walksafe_L.gif`,
  `b150_walksafe_F.gif` (medium rug, IMU filter, evaluation seed 10000).
- Other runs, kept for the record: `b150_walk_L` / `b150_walk_L2` (old reference: marched), `b150_walkflat_L` (unsmoothed
  level-sole reference), `b150_walkflats_L` (smoothed, before the stride weight), `b150_walkimu_L` (IMU-trained),
  `b150_walk_F`, `b150_walksafe_F`, and the five ski runs `b150_ski_L`, `b150_skiflat_L`, `b150_skiflat2_L`,
  `b150_skiwalk_L`, `b150_skismooth_L`.

## 7. Step-and-settle walker (`b150_step3_L`)

On the robot the walkers above fell on the first step about 70 % of the time, and the deterministic tests
(`scripts/det_walk.py`, `policy_stand.py run --run lift|sine|cpg|rock`) showed why: on one board the robot tips toward
the lifted foot almost at once. Two things from the trials shaped this walker: small quick steps worked, and a pause
after each step stops the rocking from building up.

**Reference** (`models/ref/b150L_settle.npz`, `ref_gait.py --joint-swing`): one step every 0.625 s, a 0.3 s swing and
then 0.325 s with both boards down and nothing moving. Steps are 1-3 cm. Two flat boards can only be staggered with the
front knee bent and the rear leg nearly straight (a flat sole fixes the shin angle), which puts the pelvis 38 mm ahead
of the stance pose: the "ready" pose, centre of mass 31 mm ahead of the board centres. The servos' ~35 rad/s^2 motion
profile allows only a ~10 deg knee bump in 0.3 s, so the reference lifts little by itself; the policy is paid for the
lift (`ref_swing`) and finds the rest.

**Training** (`step_b150` task; run 3 rewards `progress=4 stride=20 track_lin=1 ref_contact=0 ref_tip=0 ref_swing=3
termination=-10`, from scratch, 30 M steps, rugs down to a board sinking ~6 mm):
- episodes start in the ready pose and the forward command ramps at 0.03 m/s^2, so the first steps are millimetres
  long; a quarter of the commands are "stand", for many more starts
- `progress`: distance covered over the last gait cycle as a fraction of the command (standing earns 0)
- `ref_swing`: the scheduled foot is up during its swing (standing earns 0)
- `settle`: body rotation rate during the pause, penalized
- runs 1 and 2 (`b150_step_L`, `b150_step2_L`) used the velocity kernel and the contact / tip-height imitation terms,
  which pay standing still about half; both stood still

**Results** (simulation, IMU filter, 4 cm/s command)

| test | hard floor | medium rug | soft rug (board sinks ~6 mm) |
|---|---|---|---|
| survives 15 s (6 seeds) | 5/6 | 5/6 | 2/6 |
| forward speed | 3.6 cm/s | 3.9 cm/s | 3.2 cm/s |
| lift, air time | 6 mm, 140 ms | 4 mm, 95 ms | 6 mm, 90 ms |
| first-step test, falls in 6 s (20 starts) | 0/20 | 0/20 | 2/20 (none in the first 2 s) |

On the soft rug, the only simulated floor that reproduced the robot's first-step falls, the old walker fell 20/20
(18 within 2 s) and the safe walker 20/20. The new walker's falls are all forward, later (the ready pose's centre of
mass sits ahead). Much of its motion still slides (60-80 % of the time). GIF: `assets/gifs/b150_step_L.gif`.

On the robot (`--ready` defaults to 2 s for this walker: blend into the ready pose, then the policy starts):

```bash
python policy_stand.py run --run b150_step3_L --enable --ramp 0 --vx 0.04
```

## 8. Slower, leaned back, waits to settle (`b150_step4_L`)

Robot trials of `b150_step3_L` (2026-11-08): the next step came before the frame had settled, and the centre of mass
sat so far forward that the boards' toes stuck in the rug. Changes (task `step2_b150`, reference
`b150L_settle2`: `ref_gait.py ... --steps-per-s 1.0 --swing-frac 0.3 --lean 15 --vmax 0.06 --nv 7`):

- **one step per second**: 0.3 s swing, 0.7 s pause (was 0.3 / 0.33 s)
- **settle gate**: at each lift-off the gait clock waits while the body rotates faster than 0.3 rad/s (roll/pitch
  gyro), at most 0.5 s. Training, `CpuBiped` and the robot runner use the same rule (`settle_gate_*` in the config)
- **pelvis leaned back 15 deg**: with both boards flat and staggered, the legs' geometry fixes the hips over the feet;
  the pelvis pitch is the one free angle. Centre of mass ahead of the board centres: legs only 32 -> 21 mm (hard floor;
  ~30 mm on the soft rug, where the boards sink toe-first), full body 32 -> ~4 mm. Episodes start in this pose with
  the pelvis pitched to keep the boards level (`ref_gait.level_pitch`)

Results (IMU filter, 3 cm/s command, 15 s walks, 6 seeds; start test = 20 starts on the soft rug, falls in 6 s):

| checkpoint | hard floor: up, speed | medium rug | soft rug | start test |
|---|---|---|---|---|
| 10 M | 6/6, 2.3 cm/s | 4/6, 1.4 cm/s | 2/6, 1.1 cm/s | 1/20 |
| 20 M | 6/6, 2.0 cm/s | 5/6, 0.3 cm/s | 3/6, 0 | 0/20 |
| 30 M | 4/6, 4.4 cm/s | 5/6, 0.9 cm/s | 2/6, 0 | 1/20 |

The start is solved in simulation (no side-to-side rock at the start, 0-1 falls in 20 against 20/20 for the first
walkers). Walking on rugs is not: the boards slide 85-90 % of the time and the soft rug defeats every checkpoint.

```bash
python policy_stand.py run --run b150_step4_L --params params_0010M.pkl --enable --ramp 0 --vx 0.03   # steadier
python policy_stand.py run --run b150_step4_L --params params_0030M.pkl --enable --ramp 0 --vx 0.03   # faster on firm floors
```
