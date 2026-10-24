# Sim-to-real risks

Updated after training (2026-10-23). "Evidence" points to `docs/experiments.md`.

| risk | why it matters here | evidence from simulation | mitigation / first bench test |
|---|---|---|---|
| **Tip traction** | Ball tips must push off with no support polygon | v6b walking ≥ 1 m falls from 100 % to 67 % at μ = 0.5 and 27 % at μ = 0.35 (E17) | High-μ replaceable rubber caps; test on the target floor; measure μ |
| **Pushes / disturbances** | Micro-stepping leaves little margin | v6b: 0.3 m/s pushes cause 40 % falls; 0.5 m/s, 80 % (E17) | Train with pushes from scratch (E19 robust run); a tether is mandatory for first hardware |
| **Sensor noise** | Policy uses IMU gravity + gyro at 50 Hz | Noise ×2 drops standing to 48–66 % and walking to 80 % | Characterize the real BNO085 noise; filter; train at measured noise ×1.5 |
| Servo speed | Knee reaches 0.95–1.0 × no-load speed | E8, E17 | Bench-measure the STS3215-C046 torque-speed curve under load |
| Sustained torque / heat | RMS 0.6–0.82 × vendor "rated" (unverified figure) | E8, E17 | 10-minute stepping test with servo temperature telemetry |
| Rotor inertia (armature) | Dominates swing torque | J·N² estimate | Spin-down test; randomized 0.6–1.6× in training |
| Backlash / belt compliance | 1:1 GT2 knee belt; plastic horns | Randomized deadzone ≤ 0.9° only | Measure; consider a direct-drive knee if backlash exceeds 1° |
| Bus latency | 8 servos on half-duplex TTL | Policies tolerate 12 ms (93–96 %) | Measure the round trip at 1 Mbps; 50 Hz control is comfortable |
| Contact model | Lean contacts in training; MuJoCo soft contacts | Full-contact CPU eval agrees, but the B rocker transferred worse in v1 | Hardware is the only real test |
| Heading control | Policy needs an IMU-yaw heading-hold loop | Open-loop turn commands not followed (E10–E14) | Gyro-integrated yaw; magnetometer optional; train turning explicitly |
| Printed-shell mass | Shells are a third of total mass (estimate) | ±15 % mass randomization; +20 % mass harmless (E17) | Weigh the parts; update the config |
| Quiet standing | Point contacts never stand still | 0 % no-lift stance for the ball in every variant, including ideal control (E18, E25, E29, E38). Any flat foot stands (E24–E27); a 30 × 18 mm pad at the shin tip stands on 1 of 2 seeds (E31, E39) | Accept micro-stepping with the ball, or fit a 30 mm flat rubber pad. Bench-test real pad compliance and edge contact |
