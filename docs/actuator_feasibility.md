# Actuator feasibility — Gate B (pre-training audit)

Script: `scripts/torque_audit.py <model>` produces `results/torque_audit_<model>.json`, a
`.txt` summary, and plots in `results/plots/`. These are the numbers for baseline_B_10dof
(905 g). The other three baseline models are within ±10 % (A_8dof: knee single-stance 0.53 N·m,
hip roll 0.27 N·m).

**Scope and limits.** This audit covers:
- *quasi-static* gravity torques (exact for rigid point contacts);
- swing-leg *inverse dynamics* with the pelvis held still;
- a crouch-to-stand drive through the servo model on an explicit vertical-slider test rig.

**Dynamic stance and impact torques are not measured yet.** The ×2 "peak" column is an
assumption. RL rollouts in Gates D/E must log the real values (actuator_force, speed and
saturation over time) and replace it.

## 1. Static gravity torque (|τ|, N·m)

| pose (hip = ankle = knee/2) | knee, double support | knee, single support | hip roll, single support | stance-leg adduction needed |
|---|---|---|---|---|
| knee 10° (near straight) | 0.05 | 0.08 | 0.30 | 6.7° |
| knee 32° (stance keyframe) | 0.15 | 0.28 | 0.30 | 6.9° |
| knee 60° | 0.27 | 0.51 | 0.29 | 7.6° |
| knee 90° | 0.41 | 0.74 | 0.29 | 9.0° |
| knee 110° (deep crouch) | 0.48 | 0.86 | 0.29 | 10.8° |

Hip pitch stays below 0.1 N·m statically. Ankle and yaw are zero statically, because a point
contact transmits no moment. The ankle only matters dynamically, through rocker rolling (B).

## 2. Swing torque *and* speed vs. the full-duty torque-speed envelope

Min-jerk swing: 80 mm step length, 30° extra knee flexion, swing taking 85 % of the step period.
Torques include each set's reflected rotor inertia, J_rotor·N², which is an **estimate**.

| set (roll/pitch/knee servo) | knee peak speed / no-load @2 steps/s | max feasible step rate (knee) |
|---|---|---|
| **baseline**: STS3215-C046 @7.4 V | 0.70 | **2.5 steps/s** |
| torque_feetech_12v: STS3215-C018 @12 V | 1.64 | 1.0 steps/s |
| slim_dynamixel: XC330-T288 @11.1 V | 1.14 | 1.5 steps/s |
| xl330_only: XL330-M288 @5 V | 0.72 | 2.5 steps/s (but fails statics, below) |

At 2 steps/s the 0.08 m step gives about 0.16 m/s, the upper end of the spec's 0.05–0.20 m/s.

## 3. Peak and sustained checks (single-support stance, knee 32°)

| joint | servo | stall | vendor rated | stance τ | 2×(k ≤ 60°) / stall | stance / rated |
|---|---|---|---|---|---|---|
| hip roll | STS3215-C046 | 1.41 | 0.47 | 0.30 | 0.42 | **0.63** |
| hip pitch | STS3215-C046 | 1.41 | 0.47 | 0.04 | 0.09 | 0.09 |
| knee | STS3215-C046 | 1.41 | 0.47 | 0.28 | 0.72 | 0.59 (**1.09** at knee 60°) |
| ankle | STS3032 | 0.44 | 0.15 | ~0 | 0.03 | ~0 |
| hip yaw | STS3032 | 0.44 | 0.15 | 0 | — | dynamic only (untested) |

Derating scenarios when the vendor rating is distrusted (sustained cap = factor × stall):
0.5 / 0.33 / 0.25. At 0.25 × stall (0.35 N·m), single-support knee is feasible only at
knee ≤ ~40°. **Thermal feasibility is unverified**: there is no credible thermal model or
datasheet data. This is an open bench-test item.

Rejected sets:
- **xl330_only**: single-support knee ×2 = 196 % of stall, and hip roll exceeds its sustained cap by 1.7×.
- **torque_feetech_12v**: plenty of torque, but too slow.
- **slim_dynamixel**: knee ×2 = 111 % of stall, swing limited to 1.5 steps/s, and 3–4× the cost.
  It is the only set that meets the hip-width target, though, so it is kept as the
  "appearance-optimal" alternative.

## 4. Crouch-to-stand on the test rig (servo model, 1 kHz)

Knee 100° → 32° in 1.2 s (min-jerk), pelvis on a vertical slider.

| set | knee peak τ | rise | duty saturation |
|---|---|---|---|
| baseline | 0.41 N·m (29 % stall) | 86 mm | none |
| torque_feetech_12v | 0.56 | 86 mm | none |
| slim_dynamixel | 0.39 | 85 mm | none |
| xl330_only | 0.39 | 85 mm | none (double support only) |

## 5. Budget and electrical (planning numbers)

- Servos: 6 × STS3215-C046 (~$25) + 4 × STS3032 (~$15), about **$210** for 10-DoF and about $180 for 8-DoF.
- Bus: one Feetech TTL half-duplex bus at 1 Mbps. ESP32-S3 MCU (~$10), bus adapter (~$10), IMU BNO085 (~$20).
- Power: 2S LiPo. Stall currents are 3.3 A per STS3215-C046 and 1.2 A per STS3032, so the
  worst case of all stalling at once is ~25 A. A realistic walking draw is perhaps 3–6 A
  (unverified). 6 V buck (≥ 5 A) for the STS3032s. Bench tether first.
- Prints, bearings (MR85ZZ etc.), GT2 belts/pulleys, M2/M3 hardware, rubber caps: about $60–90.
- Planning total: **≈ $350–420** (inside the $250–600 band).
- Latency: 10 servos × (write + read) at 1 Mbps is ~2–4 ms per cycle (estimate), so a 50 Hz
  policy is comfortable and 100 Hz is plausible. Needs a bench test.

## 6. Open verification tasks
- Bench-measure the STS3215-C046 torque-speed curve and thermal rise at 0.3–0.5 N·m continuous.
- Measure rotor inertia and backlash (the reflected inertia J·N² drives swing torque).
- Replace the ×2 stance-peak assumption with logged RL-rollout torques.
- Check that the hip-yaw STS3032 (0.44 N·m stall) is adequate under turning.

## 7. Measured in trained policies (replaces the ×2 assumption)

CPU evaluation, full-contact model, servo model in the loop, median over episodes (peak = max over all episodes). Left leg shown; right is within ±0.03.

| policy | metric | hip yaw | hip roll | hip pitch | knee |
|---|---|---|---|---|---|
| stand v2 (A), 30 s | RMS τ [N·m] | 0.08 | 0.40 | 0.38 | 0.41 |
| | RMS / vendor rated | 0.53 | 0.84 | 0.80 | 0.88 |
| | peak τ / stall | 0.89 | 1.00 | 1.00 | 1.00 |
| | peak speed / no-load | 0.72 | 0.87 | 0.94 | 1.00 |
| | duty-saturated fraction / over-speed events | 0.007 / 1 | | | |
| walk v3 (A), 0.15 m/s | RMS τ [N·m] | 0.08 | 0.47 | 0.42 | 0.40 |
| | RMS / vendor rated | 0.55 | 0.99 | 0.90 | 0.84 |
| | peak τ / stall | 1.00 | 1.00 | 1.00 | 1.00 |
| | peak speed / no-load | 0.76 | 0.98 | 0.97 | 1.06 |
| | duty-saturated fraction / over-speed events | 0.016 / 10 | | | |
| walk v3 start/stop | RMS τ [N·m] | 0.08 | 0.46 | 0.41 | 0.39 |
| | RMS / vendor rated | 0.55 | 0.97 | 0.88 | 0.83 |
| | peak τ / stall | 1.00 | 1.00 | 1.00 | 1.00 |
| | peak speed / no-load | 0.67 | 0.98 | 0.93 | 1.18 |
| | duty-saturated fraction / over-speed events | 0.014 / 7 | | | |
| stand v1 (A, no torque penalty) | RMS τ [N·m] | 0.09 | 0.61 | 0.50 | 0.46 |
| | RMS / vendor rated | 0.58 | 1.29 | 1.06 | 0.99 |
| | peak τ / stall | 1.00 | 1.00 | 1.00 | 1.00 |
| | peak speed / no-load | 0.66 | 1.09 | 0.97 | 1.29 |
| | duty-saturated fraction / over-speed events | 0.041 / 85 | | | |

Reading:
- With the v2 hardware-plausibility terms, sustained (RMS) torque stays at **0.53–0.88 × the vendor
  rated torque** on every joint when standing. Walking pushes hip roll to **0.99 × rated**, right at
  the limit. Without those terms (v1), hip roll
  runs at **1.29 × rated**. Plausibility comes from the reward, not for free.
- Peak torque reaches stall on most joints in every policy. These are short impact/recovery spikes
  (duty saturated 1–2 % of the time). They pass the permissive peak check by definition (the model
  clips at stall) and are logged, not hidden.
- Joint speed reaches ~0.9–1.0 × no-load on the knee. Speed, not torque, is the binding limit,
  consistent with the Gate B finding that rejected the slower 12 V C018.
- Thermal behaviour remains **unverified**. Running at 0.8–0.9 × a vendor "rated" figure that we
  could not confirm from an official datasheet is not a thermal guarantee.

### Selected policy `walk_A8_v6b_ft` (40 s walk at 0.15 m/s, 100 seeds)

| metric | hip yaw | hip roll | hip pitch | knee |
|---|---|---|---|---|
| RMS τ [N·m] | 0.10 | 0.38 | 0.30 | 0.33 |
| RMS / vendor rated | 0.65 | 0.81 | 0.64 | 0.69 |
| peak τ / stall | 0.96 | 1.00 | 1.00 | 1.00 |
| peak speed / no-load | 0.74 | 0.95 | 0.85 | 0.88 |

Duty saturated 0.4 % of the time; 2 over-speed events in 100 episodes (10 servo substeps × 2000 control steps each). Passes the vendor-rated sustained check on every joint and the permissive peak check. It would fail the conservative 0.25 × stall scenario on hip roll and knee (RMS ≈ 0.27–0.29 × stall).
