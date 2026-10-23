# Morphology — Gate A (appearance review)

Status: **primary design selected 2026-10-22: baseline, foot A (ball tip), 8 DoF.**
Foot B and the ankle servo stay as ablations. No training has been run yet.

Joint axes and ranges: `assets/renders/baseline_A_8dof/joint_axes.png`. Each joint swept through
its range: `assets/renders/baseline_A_8dof/joint_motion.png` (`scripts/joint_diagram.py`).

All numbers below are *measured from the compiled MuJoCo model* in the `stand` keyframe
(`scripts/inspect_model.py` → `results/morphology_*.json`), not copied from the config.

## Frames and conventions

- World: Z up, X forward, Y left. Pelvis origin = centre of the pelvis shell.
- Chain per leg: pelvis → hip_yaw (Z) → hip_roll (X) → hip_pitch (Y) → knee (Y) → [ankle (Y)] → tip.
- Signs (mirrored axes, so both legs share them): +yaw = toe out, +roll = abduction,
  +hip_pitch = flexion (foot forward), +knee = flexion, +ankle = dorsiflexion (toe up).
  Verified in `tests/test_model_integrity.py::test_joint_signs`.
- Stance keyframe: hip 16°, knee 32°, ankle 16° (tip level).

## Three geom layers

| layer | group | collides | mass |
|---|---|---|---|
| exterior shells (convex-hull meshes) | 0 | no | printed-shell mass, *shell inertia* |
| servo case envelopes (boxes) | 1 | no | catalog servo mass |
| collision primitives | 3 | yes | 0 |
| point masses (electronics, fasteners, wiring, belts) | 4 | no | itemised |

The rubber tip is both the visible foot and the **only** collision geom below the shin
(`tests/test_contacts.py` checks it is the lowest colliding geom; no hidden sole).

## Baseline numbers (baseline_B_10dof)

| quantity | value | target | |
|---|---|---|---|
| body height (shell top, stance) | 377 mm | — | 397 mm incl. battery pod |
| pelvis origin height | 344 mm | 300–350 mm | ✓ |
| hip width (pelvis + hip covers) | 107 mm = **28.4 %** | ≤ 24–28 % | ✗ marginal (see below) |
| shin max visible width (front) | 22 mm = **5.8 %** | ≤ 9–12 % | ✓ well under |
| shin section at the calf | 22 × 36 mm (superellipse n = 3.5) | | |
| thigh max width | 44 mm = 11.7 % | — | |
| knee cover width | 29 mm | | |
| thigh / shin length | 138 / 150 mm (shin : thigh = 1.09) | 130–155 / 140–165 | ✓ |
| hip-yaw axis spacing | 60 mm | 70–90 mm | narrower than the hypothesis |
| total mass | 905 g (10-DoF), 858 g (8-DoF) | 0.6–1.0 kg | ✓ |
| CoM height at stance | 260 mm | | |
| packaging | all 10 servo cases inside shells with ≥ 1.2 mm wall clearance | | ✓ |

Mass split (10-DoF): servos 410 g · printed shells 306 g · electronics/fasteners/wiring/belts 177 g · rubber tips 12 g.
Per body: pelvis 176 g, hip-yaw links 2×90 g, hip-roll links 2×90 g, thighs 2×124 g, shins 2×53 g, feet 2×7 g.

### Feet

| | A: point tip | B: micro-rocker |
|---|---|---|
| geometry | rubber sphere, r = 8 mm | ellipsoid 32 × 13 × 15 mm |
| sagittal contact radius | 8 mm | 34 mm (a²/c) |
| contact-point travel over ±20° of tip pitch | **0 mm** | 20 mm |
| rigid static support | single point | single point |

Note for A with an ankle: rotating a sphere about its own centre-adjacent axis does not
move the contact point, so **ankle pitch has ~zero centre-of-pressure authority with foot A**.
The A/B comparison should therefore be run primarily at 8 DoF (identical topology,
fixed tip), with 10 DoF as an ablation where the ankle is meaningful only for B.

## Why the hips are the widest/boxiest part

The 45.2 × 24.7 × 35 mm STS3215 case sets the hip:
- The hip-roll servo hangs below the yaw joint (shaft along X). Its 45 mm length fixes
  yaw→roll = 37 mm.
- The hip-pitch servo must sit below the roll servo without overlapping it, so
  roll→pitch = 26 mm. Its body extends 34 mm **behind** the pitch axis. That is the
  backward "shelf" visible in the side view.
- The pitch servo's 35 mm shaft-axis depth plus the horn and wall gives a 47 mm hip cover.
  With 60 mm axis spacing, the hips measure 107 mm (28.4 %).

What was tried to slim it (recorded in `docs/experiments.md`):
- The yaw servos point inward instead of forward, so the pelvis depth went from 66 to 54 mm.
- The hip spacing went from 64 to 60 mm.
- The knee servo moved to the upper thigh with a hidden 1:1 GT2 belt (`knee_drive: remote`).
  The thigh then tapers naturally into a 29 mm knee instead of bulging to 45 mm at the knee
  (compare the `direct_knee` variant).

## Geometry variants (Gate A sketches)

| variant | change | hip % | shin % | thigh | mass | packaging |
|---|---|---|---|---|---|---|
| baseline | — | 28.4 | 5.8 | 44 | 905 g | ✓ |
| long_shin | thigh 130, shin 170 | 27.5 | 5.7 | 44 | 906 g | ✓ |
| direct_knee | knee servo at the knee | 28.4 | 5.8 | 44 (knee 45) | 904 g | ✗ 1.0 mm short of wall clearance |
| slim_dynamixel | XC330/XL330 (34 mm cases) | **24.0** | 8.2 | 36 | 665 g | ✗ yaw case touches shell; ~4× servo cost |

## Open review questions

1. Is a 28 % hip width with a dark roll-joint band acceptable, or should we pay for the
   slim (XC330) hip?
2. Is the side-view hip shelf (pitch-servo body behind the axis) acceptable?
3. Squarish thigh section (n = 4) vs rounder (it needs about 4 mm more width).
4. Long-shin proportions vs baseline.
5. Shell colours / dark joint covers / red accent band / battery pod: keep or change.
