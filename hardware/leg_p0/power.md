# Power and wiring: two-leg robot, 8 × STS3215

Estimated servo current and power for the as-built two-leg robot:
- 6 × STS3215-C001 (1:345) on yaw, roll and pitch;
- 2 × STS3215-C044 (1:191) on the knees;
- P08 pelvis, 80 mm hips;
- a 7.4 V supply.

The numbers come from trained policies running in the CAD-derived simulation (`models/hw_p0_8dof`,
docs/experiments.md round 3). `scripts/power_audit.py` turns each servo's simulated torque, speed and
PWM duty into supply current using the same DC-motor model the simulation uses.

## Summary

| activity (policy) | mean | 99th percentile | peak (20 ms average) | peak (2 ms) | mean power |
|---|---|---|---|---|---|
| quiet stand (`hw_stand_ft`) | **0.4 A** | 0.9 A | 5.9 A* | 6.6 A* | 3 W |
| quiet stand, push-trained (`hw_stand_push`) | 0.7 A | 5.2 A | 10.0 A | 12.5 A | 5.5 W |
| standing on a stepping policy (walkers at zero command) | 1.7–1.9 A | 5.0–5.5 A | 7–9 A | 9–11 A | 12–14 W |
| **walking, best walker** (`hw_walk_scratch_stride2`, 0.15 m/s command) | **1.5 A** | **4.7 A** | **8.9 A** | 10 A | **11 W** |
| … same walker at 0.05 / 0.10 / 0.20 m/s commands (`hw_walk_scratch_stride`) | 0.7 / 1.1 / 1.6 A | 2.6 / 4.0 / 5.4 A | 8.9–9.6 A | ≤ 10.5 A | 5 / 8 / 12 W |
| walking, warm-started walkers (round-2 gait transferred, 0.07–0.12 m/s) | 1.8–2.2 A | 5.6–6.4 A | 7.7–9.2 A | 9.9–11.7 A | 13–16 W |
| walking, shuffle gait (`hw_walk_scratch`) | 1.1 A | 3.5 A | 8.8 A | 9.5 A | 8 W |
| **walking with real steps, round 4** (`r4j_d65_c4_cont`, 0.15 m/s command) | 2.15 A | 6.5 A | 9.0 A | 13 A | 16 W |
| … catwalk style (`r4k_cat2_ft`) / 60 mm hips (`r4n_s60_d65`) | 2.2 / 2.0 A | 6.8 / 6.5 A | 14.6† / 9.0 A | 15.6 / 12.7 A | 16 / 15 W |
| all 8 servos stalled at once (bound, never seen) | | | 22.6 A | | 167 W |

\* The quiet-stand peaks happen while the robot settles in the first second after the start pose.
† One rare event in 20 episodes; 99 % of 20 ms windows stay below 5.5 A.

**Round 4 (real steps instead of the skitter, docs/experiments.md E46–E56) costs about 45 % more current:** ~2.2 A
mean (16 W) instead of ~1.5 A, because every step lifts the leg ~2 cm and swings it ~6 cm. The hip-roll and
hip-pitch servos run at their rated torque (RMS ≈ 1.0 × rated, vs 0.83 for the skitter) and this did not come down
with stronger torque penalties or 60 mm hips: **expect the hips to get warm in long walking sessions**; watch the
servo temperature register and walk in bouts.

How much walking costs depends on the gait more than on the speed. The best walker (learned natively on
the hardware model) scales with speed: 0.7 A at 0.05 m/s up to 1.6 A at 0.2 m/s, and it stops and stands at
0.55 A at zero command. The warm-started walkers draw 1.9–2.0 A at every speed from 0.05 to 0.2 m/s: they step
about 5 times a second regardless, including while "standing". A policy that steps in place costs about 4× the
power of a real quiet stance. **The peaks (~9 A over 20 ms) are the same for every gait**, so size the wiring
for them either way.

Per servo, walking:

| servo | mean | RMS | 99th percentile | peak (= stall current) |
|---|---|---|---|---|
| hip yaw (C001) | 0.10–0.14 A | 0.2 A | 0.6–0.7 A | 1.1 A |
| hip roll (C001) | 0.2 A | 0.3 A | 1.2–1.4 A | 2.5 A |
| hip pitch (C001) | 0.2 A | 0.35–0.4 A | 1.2–1.3 A | 2.5 A |
| **knee (C044)** | **0.36 A** | **0.7 A** | **2.9–3.2 A** | **3.8 A** |

Every hip and knee servo reaches its stall current briefly. The knee draws the most: it is at more than 80%
of stall current about 1% of the time while walking. Its RMS torque is about 0.8 × rated (1.0 × rated if the
real knee is at the pessimistic end of its torque estimate), so **knees will run warm in long walking sessions**.

## What to size for

| item | recommendation | why |
|---|---|---|
| **supply** | 7.4 V, **≥ 10 A peak** (15 A gives margin for rare round-4 transients), ≥ 3 A continuous | walking: 2 A mean, ~9 A peaks over 20 ms. A bench supply in CC mode should be set to 8–10 A, not 3 A, or it will sag on every step. |
| **5 V bench supply (now)** | fine for setup and slow routines | the servos run slower and weaker on 5 V; a walking policy trained for 7.4 V won't behave the same on it |
| **bulk capacitor at the robot** | 1000–2200 µF, ≥ 16 V, low-ESR, across the servo power input near the pelvis | absorbs the 2 ms peaks (10–12 A) that the tether wire can't deliver without dropping the voltage |
| **tether** (supply → robot) | 18 AWG (or thicker) pair, ≤ 1.5 m | at 10 A peaks, 1.5 m of 18 AWG (both conductors, ~0.06 Ω) drops ~0.6 V; 22 AWG would drop ~1.5 V |
| **power distribution** | **one branch per leg** from the pelvis, not one daisy chain for all 8 | each STS lead and its 3-pin connector (~3 A per contact, Molex 5264-style) carries every servo downstream of it. One leg's 4 servos reach ~4–5 A in 20 ms peaks. A single chain through all 8 servos would put ~9 A through the first connector. |
| **in a leg** | board → yaw → roll → pitch → knee is OK if each leg has its own feed | the first link carries ~2 A RMS / ~5 A peak, the knee link only its own servo |
| **bus adapter board** | check its power-path rating (screw terminal or DC jack) against ~10 A peaks | some small adapter boards route servo power through a barrel jack or thin traces rated far lower |
| **battery (later)** | 2S LiPo; 450 mAh ≈ 15–18 min of walking, 1000 mAh ≈ 35–40 min (best walker) | 11 W walking (14–16 W with a step-in-place gait). Peaks of 10 A need ≥ 20C at 450 mAh (≥ 10C at 1000 mAh) |

## Supply voltage choice

A full 2S pack is 8.4 V, not 7.4 V. The simulation with the servos scaled to 8.4 V (+13% speed and torque,
`models/hw_p0v84_8dof`) walked **46% faster** (0.111 vs 0.076 m/s) and stopped on command twice as often
(50% vs 26%), at about the same current (1.9 A mean, 9.2 A 20 ms peak), so ~16 W at 8.4 V. The
as-built servos are speed-limited, so the extra 13% matters. 8.4 V is the top of the
LiPo range and above the servos' 7.4 V nominal rating, so check Feetech's absolute maximum for your units before relying on it.
A bench supply at 8.0–8.4 V gets the same benefit for testing.

## How the numbers were made, and their limits

- Motor current I = I_stall × τ / τ_stall from the simulated output torque, with the vendor's stall currents:
  2.5 A for the C001 (2.0 A @ 6 V, scaled), 3.8 A for the C044.
- Supply current = PWM duty × motor current while the motor is driving (duty and current with the same sign).
  Regenerated (braking) current is ignored, which errs on the safe side.
- 30 mA per servo is added for its own electronics.
- **Not modelled:** gearbox friction losses (real currents are somewhat higher), MCU/IMU (add ~0.1–0.2 A),
  and the servo firmware's own current limit and PID (real peaks may be clipped lower).
- Physics runs at 2 ms, so "2 ms peak" is the finest resolution. True inrush spikes are shorter and higher,
  which is what the capacitor is for.
- Measure on the real robot: an inline power meter on the tether during a `routine step` run gives the mean and
  20 ms peak to compare with this table.

Regenerate: `.venv/bin/python scripts/power_audit.py <run> --params <ckpt> --test walk40|stand --n 20 [--volts 7.4]`
(writes `results/power/<run>__<test>.{json,png}`).
