# Sim to real: from the first fall to recovery steps

How the stand policies went from falling within a second on the robot to stepping in reaction to a fall, and what
each phase taught us (2026-11-06 to 11-07). The servo measurements and the first round of retraining are covered in
depth in [servo_identification.md](servo_identification.md); this doc covers the whole arc.

**Status.** The robot (legs and pelvis only, no torso or head, 5 V bench supply, IMU on the pelvis) now stands on its
own for minutes once it is in balance, and with fast hip-pitch servos it steps in reaction to a fall. Whether those
steps actually catch it is still open. Only one of the two swapped hip-pitch servos turned out to be fast (§3.7).

## 1. Setup

| | |
|---|---|
| Robot | two P0 legs on the P08 pelvis, 8 × STS3215, torso and head off |
| Supply | 5 V bench supply (the servos are 7.4 V parts; 5 V gives ~0.68 of rated torque and speed) |
| IMU | MPU6500 on an RP2040-Zero, taped to the hip plate, streaming gravity and gyro at ~92 Hz with ~5 ms latency ([imu_stream.py](../hardware/imu/imu_stream.py)) |
| Runner | [policy_stand.py](../hardware/leg_p0/software/policy_stand.py): the trained policy in numpy at 50 Hz, safety stop past 25° tilt, every run logged to `stand_<run>_<time>.csv` |
| Policy | MLP, 35 inputs (gravity, gyro, joint angles and speeds, last action, command, clock) → 8 joint targets around the stance (hip 16°, knee 32°) |

## 2. Phases and the policies kept for each

Weights for each of these are in `runs/` (one checkpoint each), so the phases can be filmed side by side.

| # | policy | trained on | what it does on the robot |
|---|---|---|---|
| 1 | `hw_stand_push2` | the original servo model (spec sheet, no firmware profile) | falls within ~1 s; fights a supporting hand |
| 2 | `hw_stand_fit5v_p005` | servo model fitted to step tests, light shoves | feet stay planted; never run on the robot with its feet free (simulation only) |
| 3 | `hw_stand_fasthip5v` | fitted model, but hips assumed as fast as the knees | stands quietly for minutes once balanced (143 s and 301 s); falls when disturbed, because it starts steps the slow hips can't finish |
| 4 | `hw_recover_slowhip` | fitted model, stepping allowed, random placements and shoves | simulation only: recovery barely improves with the real hips (comparison baseline) |
| 5 | `hw_recover_p191` | as 4, with both hip-pitch servos swapped to 1:191 | steps in reaction to falls, sensibly; whether the steps catch it is still open |

Fine-tune chain (weights kept too): `push2` → `fit5v_p01` → {`fit5v_p005`, `fasthip5v`, `fit5v_step` →
`recover_slowhip`}; `fasthip5v` → `recover_p191`.

Run any of them from `hardware/leg_p0/software`:
```bash
python policy_stand.py run --run <policy> --enable --pitch-trim 2
```
`hw_recover_p191` needs the swapped hip-pitch servos; the others expect the original 1:345 hips.

## 3. What we learned

### 3.1 The servo firmware's motion profile was the big gap

The first policy fell within a second. Step tests (`policy_stand.py step-test`) showed every servo following an
internal constant-acceleration ramp toward each new goal: ~400°/s² on the 1:345 hips and ~1900°/s² on the 1:191 knees
(with the acceleration register at its maximum, 254). A 5° correction on a hip takes ~250 ms. The published specs
(stall torque, no-load speed) say nothing about this, and the original simulator modelled none of it.

A servo model with that profile, ~10 ms of dead time and the measured gain, fitted to the step tests
([fit_servo.py](../scripts/fit_servo.py)), matches the hips to 0.68° RMS, against 3.1° for the original model.

### 3.2 The original policy was fragile in exactly the ways real hardware differs

On the fitted model, the original policy still stood 75 % of the time, but 1° of gear backlash dropped that to 6 %
([gap_probe.py](../scripts/gap_probe.py)). Policies retrained on the fitted model with randomized profile, delay,
voltage and backlash shrugged off the same perturbations: 96 % quiet standing.

### 3.3 Train from the start the robot actually gets

The robot engages from a commanded stance, tilted ~1°. The original training reset (joints off by ~5°, tilt ~2°) was
so harsh with slow servos that most episodes ended before the policy saw a shove. Matching the reset to the hardware
made training ~3× faster.

### 3.4 The balance point and the IMU zero matter by single degrees

Without ankles, a 30 mm pad leaves little margin: the centre of mass sits ~5 mm ahead of the pad centre, and each
degree of forward lean moves it ~5 mm, so ~2° forward is already past the pad's edge. In the first trials the IMU
read +3 to +7° at engage, so the robot was already falling when the policy took over.

Standing quietly, the robot balanced at an IMU pitch of +1.7 to +3.4°, against +0.8° in simulation: the IMU zero is
1–2.5° off the policy's idea of upright. `--pitch-trim` subtracts a fixed offset; 2° is the current value.

### 3.5 Feet-planted standing can't recover; stepping needs fast hips

Without stepping, the most this robot can absorb is about √(g/h) × the reach of the pad. That's ~0.05–0.09 m/s
fore-aft, a gentle nudge. Beyond that it has to step.

`hw_stand_fasthip5v` had learned to step, but for hips as fast as the knees. On the robot, its left and right
hip-pitch commands were more than 4° apart 40–66 % of the time while disturbed, and almost never while quiet. In the
fast falls it started a step ~0.4 s in and fell during it ([step_probe.py](../scripts/step_probe.py)). The
feet-planted policy `fit5v_p005` steps almost never in simulation (0.4 % of the time against 5.5 %).

### 3.6 More training doesn't fix slow hips; faster hips do

Recovery training (stepping allowed, random placements, 0.12 m/s shoves) on the same settings, with the hips as they
are and with both hip pitches swapped to 1:191. Each number is the share of 12 simulated runs still standing at the
end ([recover_eval.py](../scripts/recover_eval.py)), at 5 V:

| test | current hips (`recover_slowhip`) | 1:191 hip pitch (`recover_p191`) |
|---|---|---|
| placed leaning forward 2° / 4° | 50 % / 0 % | **92 % / 17 %** |
| placed leaning back 4° / 6° | 67 % / 8 % | **83 % / 25 %** |
| sudden forward speed 0.15 / 0.2 m/s | 0 % / 0 % | **100 % / 92 %** |
| sudden backward speed 0.15 / 0.2 m/s | 33 % / 0 % | **100 % / 83 %** |
| repeated shoves 0.05 / 0.1 / 0.15 m/s | 75 % / 17 % / 0 % | **100 % / 83 % / 25 %** |

With the current hips, training plateaued (episode length 270 → 301 of 1000 over 40 M steps). With the swap it
reached 658. At 7.4 V the swap does slightly better (100 % at 0.1 m/s shoves); the current hips gain nothing, so the
supply isn't the limit.

The 1:191 servo trades ~20 % of stall torque (1.5 against 1.9 N·m at 7.4 V; a retailer figure) for ~1.7× the speed and the
fast profile. Simulation found that enough for standing and stepping, including at 5 V.

### 3.7 Only one of the swapped servos is fast

Step test after the swap (in the air, 5° steps):

| | ramp | moved at 100 ms | 90 % reached |
|---|---|---|---|
| left hip pitch (new) | ~1650°/s² | 3.5° | 124 ms |
| right hip pitch (new) | ~420°/s² | 0–0.4° | 269 ms |
| old 1:345 hips (for reference) | ~400°/s² | | ~250 ms |

The right one behaves exactly like the old hips. Not yet known: a 1:345 unit among the spares, or the same servo
with a different firmware or EEPROM default. That needs the servo's label, how hard it back-drives by hand, and its
registers compared with the left one. The first recovery trials ran with one fast and one slow hip pitch.

### 3.8 Smaller lessons

- **7.4 V barely helps.** The firmware profile, not motor power, sets how fast a correction starts.
- **A non-learning balancer** (IMU tilt fed back to the hips, gains grid-searched, `--run balancer`) adds little to a
  plain stance hold and does worse under shoves: without ankles, the hips have little authority over where the weight
  sits on the pads.
- **The acceleration register** (41) sets the profile on the knees and the left new hip pitch, but not on the old
  hips. The runner writes 254.
- **The quiet-stance yaw** (~6° toe-out held constantly) is learned behaviour; simulation does the same.
- **IMU plumbing:**
  - Hardware I2C failed on this wiring and 400 kHz software I2C crashed, so the board runs software I2C at 100 kHz,
    with retries.
  - Reading the serial stream in big blocks added 27 ms of latency; reading what's waiting cut it to 5 ms.
  - Tape doesn't hold the board steady, so there's now a printable adapter slab
    ([imu_slab.py](../hardware/imu/cad/imu_slab.py)).

## 4. Robot trials so far

All with a hand near or on the robot, so these numbers are indicative, not a benchmark:

| policy | runs | time on robot | median run | longest | runs ≥ 10 s |
|---|---|---|---|---|---|
| `hw_stand_push2` | 24 | 2.3 min | 2.3 s | 19 s | 6 |
| `hw_stand_fasthip5v` | 23 | 14.2 min | 3.5 s | 434 s | 9 |
| `hw_recover_p191` (one fast hip pitch) | 22 | 14.2 min | 9.5 s | 230 s | 11 |

A run ends at the 25° safety stop or when the operator stops it.

## 5. Next steps

1. **Find out why the right hip pitch is slow.** Check its label and how hard it back-drives by hand, and compare its
   registers with the left one. Replace it if it's a 1:345.
2. **Bolt the IMU onto the slab**, recalibrate, and read the new balance point from a quiet stand.
3. **Refit and retrain once both hip pitches are fast.** Step-test them at 7.4 V, refit the servo model, and fine-tune
   `recover_p191` on the measured profile.
4. **Hip roll:** sideways recovery needs fast hip-roll servos too (two more 1:191).
5. **Torso and head** only after recovery works without them; their mass needs retraining.

## 6. Tools

| script | what it does |
|---|---|
| [policy_stand.py](../hardware/leg_p0/software/policy_stand.py) | runs a policy on the robot; `imu-calib`, `imu-check`, `step-test`, `timing`, `reset-gains` |
| [fit_servo.py](../scripts/fit_servo.py) | fits the servo model (profile, dead time, gain) to step-test logs |
| [gap_probe.py](../scripts/gap_probe.py) | one-at-a-time robustness probes (delay, backlash, gain, friction) |
| [lean_test.py](../scripts/lean_test.py) | standing after a fixed initial lean |
| [eval_fitted.py](../scripts/eval_fitted.py), [pd_balance.py](../scripts/pd_balance.py) | stand policies and the balancer on the fitted model |
| [step_probe.py](../scripts/step_probe.py) | how often a policy steps, in simulation and in robot logs |
| [recover_eval.py](../scripts/recover_eval.py) | recovery tests: tilted placement, sudden speed, repeated shoves |

Simulation model of the swap: `models/hw_p0p191_8dof*` (actuator set `hw_p0_c044_pitch`). `prof_acc` takes an
optional third entry for the hip-pitch profile.
