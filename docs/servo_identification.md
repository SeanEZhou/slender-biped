# First hardware stand tests, servo identification and retraining (2026-11-06)

The first time a simulation policy ran on the assembled robot (legs + pelvis, no torso or head, 5 V bench supply) it
fell within about a second, usually backward. This note records why, what was measured, how the simulator's servo
model was corrected, what retraining on it gives, and how a non-learning balancer compares. Each result is marked
**measured** (on the robot), **simulated**, or **not yet tested on hardware**.

**Summary**

* The real servos respond to small corrections 4-9x later than the simulator's servo model assumed: their firmware
  drives each move along a slow motion profile (hips ~400 deg/s^2), which the published specs don't describe.
* A servo model fitted to step tests matches the measured joint trajectories 4x better (0.68 vs 3.1 deg RMS on the
  hips). On it, the original policy is fragile in exactly the ways a real robot differs (1 deg of gear backlash takes it
  from 75 % to 6 % of quiet 20 s stands).
* Policies retrained on the fitted model stand quietly in 96 % of randomized simulated runs and are insensitive to
  backlash, extra delay and gain errors; they ride out ~0.05 m/s shoves about three times in four. **Not yet tested on
  the robot.**
* Past ~2 deg of forward lean the robot's weight is already beyond the front of its pads; the robot read +3 to +7 deg
  when the policy engaged in the first trials. Check the IMU zero and engage only when upright.
* 0.1 m/s fore-aft shoves are beyond the robot on its current hips (about 5 % survive, for every controller) and
  beyond a non-learning hip balancer at all. **With knee-speed hip servos a retrained policy survives them 62 % of the
  time**: the hip servos' motion profile is the hardware lever. A 7.4 V supply alone barely changes it.

## 1. What happened on the robot (measured)

`hardware/leg_p0/software/policy_stand.py run --enable` ran `hw_stand_push2` (the best as-built stand policy: 98 %
upright with shoves in the original simulator) at 50 Hz: IMU on an RP2040 (gravity + gyro, ~5 ms old), one sync read
and one sync write of all 8 servos per step (~2 ms). The runner's observation was checked against the simulator's to
0.0 over 300 steps, and the IMU's tilt and gyro axes against each other, so the policy received correct inputs.

24 runs (`stand_*.csv` logs): with a hand assisting, the policy "fought" it; let go, it fell in ~1.1 s, mostly
backward. In the logs the policy answers a backward tip with large left/right-alternating hip pitch commands, and the
joints arrive late. Cross-correlating command and measured angle over all runs:

| joint | lag of joint behind command, robot | same policy in simulation |
|---|---|---|
| hip yaw / roll / pitch | 0.16 - 0.38 s | 0.02 - 0.04 s |
| knee | 0.06 - 0.07 s | 0.04 s |

## 2. Servo step tests (measured)

`policy_stand.py step-test` steps one joint at a time (+A, 0, -A, 0 from the stance) and records it at ~500 Hz, with
the robot held in the air or standing. All at 5 V, servo P gain 32 (factory). Logs: `steptest_*.csv`.

* **Not load.** In the air the hips needed 90-120 ms to reach 10 % of a 5 deg step and 210-300 ms to reach 90 %;
  standing added only 10-30 ms.
* **The servo firmware's motion profile.** Every step starts as a clean constant-acceleration curve (fit residual
  ~2 ms): the servo's internal target ramps toward the goal. Hips: 180-530 deg/s^2. Knees at acceleration register
  254: ~1900 deg/s^2. The average speed scales as sqrt(step size) (5 -> 15 deg: x1.8 measured, x1.73 predicted), as
  a fixed-acceleration profile does; a speed cap or a motor limit would not.
* **Acceleration register (41).** 0 means "no ramp" on the new left knee (4760 deg/s^2, 6 ms start) but a slow default
  ramp (~270 deg/s^2) on the older right knee. 254 (the maximum) made both knees fast (~1900 deg/s^2, delay 54 ms
  instead of 110 ms). The hips did not change at all: ~400 deg/s^2 either way. The runner now writes 254.
* **P gain (21).** 32 -> 96 on four hips: 20 % faster on two, slightly slower on the other two. Not the limit.
* **Hips vs knees.** All servos report firmware 3.10 and the same model number. The knees (C044, 1:191) cruise about
  twice as fast as the hips (C001, 1:345), matching the gear ratio. Several EEPROM defaults differ between the two
  (phase byte 12 vs 92, minimum startup force 16 vs 0, protection current 500 vs 300); none was changed.

## 3. Fitted servo model (simulated)

The simulator's servo (shared by `slender_biped/mjx_env.py` and `evaluation.py`) gained three optional parts, all off by
default so older runs evaluate identically (checked: same actions to 2 decimals):

* `prof_acc` (hip, knee): the internal target follows a motion profile (accelerate at `prof_acc`, brake to stop on the
  command, speed capped at `slew`) instead of moving at 8 rad/s;
* `delay_steps`: command dead time in whole control steps, on top of the existing 0-12 ms sub-step latency;
* `speed_scale` with the existing `strength_scale`: the supply (5 V on 7.4 V servos ~ 0.68). `dr_volt`,
  `dr_prof_acc`, `dr_delay_steps` randomize them in training.

`scripts/fit_servo.py` replays every step test in the CPU simulator (pelvis held 15 cm up for the air tests) over a
grid of dead time x profile acceleration x kp, at 5 V. Result (`results/servo_fit.json`, plot `results/servo_fit.png`):

| | hips | knees |
|---|---|---|
| dead time | 10 ms | 10 ms |
| profile acceleration | **7 rad/s^2 (~400 deg/s^2)** | **35 rad/s^2 (~2000 deg/s^2)** |
| kp | 8 (unchanged) | 8 |
| RMS trajectory error, fitted | 0.68 deg | 0.26-0.39 deg (5 deg steps) |
| RMS error, the model the policies were trained on | 3.1 deg | 1.2 deg |

The 5 V scale alone hardly helps (3.1 -> 2.85 deg on the hips): the motion profile is what matters. The hip optimum is
flat (acc 5-9, dead time 0-30 ms, kp 8-16 all within 0.02 deg). One 15 deg right-knee test fits only a much slower
profile and is slower than the same servo's 5 deg steps; it is treated as an outlier (likely contact while hand-held)
and covered by randomizing the profile x0.6-1.5 in training.

**How well the fitted model reproduces the robot** (`scripts/gap_probe.py`; quiet stance, hardware-like start, 1 s
blend-in as on the robot, 16 runs of 20 s, no randomization). On the robot `hw_stand_push2` fell in ~1.1 s every time.

| added to the fitted 5 V model | `hw_stand_push2` | `hw_stand_fit5v_p005` | `hw_stand_fasthip5v` |
|---|---|---|---|
| nothing | 75 % standing | 81 % | 88 % |
| 1 deg gear backlash | **6 %** (falls at 2.0 s) | 81 % | 81 % |
| 20 ms sensor delay + 40 ms dead time | 31 % (2.0 s) | 81 % | 81 % |
| servo kp halved | 38 % (2.8 s) | 81 % | 81 % |
| hip profile x0.6, 60 ms dead time, pad friction 0.4 (each) | 50-69 % | 81 % | 81-88 % |

The fitted model alone is still more forgiving than the robot (75 % vs falling every time), but the original policy
collapses as soon as ordinary hobby-servo imperfections are added; the retrained ones don't notice them. The remaining
gap to the robot is most likely a mix of backlash and delays. (An earlier check that reported "20/20 falls at 0.8 s"
for the fitted model was contaminated by the policy's own 0.3 m/s training shoves and a harsher start; superseded.)

## 4. Retraining on the fitted model (simulated)

All runs fine-tune on the `stand_quiet2` task (stand still, both pads down, liftoff penalized) with the fitted servos
randomized around the fit: profile acceleration x0.6-1.5 per joint, command delay 4-40 ms, supply 0.60-0.75 (5 V
with sag) or 0.90-1.05 (7.4 V), on top of the original strength/speed/gain/backlash (up to 0.9 deg), mass and friction
randomization. 60M steps each, ~30 min on the RTX 4070 SUPER (~40k steps/s). Queue: `runs/queue_done.txt`.

| run | from | start pose | shoves | other | training episode length (of 1000) |
|---|---|---|---|---|---|
| `hw_stand_fit5v` | `push2` | original reset (4.6 deg/joint, 2.3 deg tilt, 1 sigma) | 0.3 m/s | | 60 -> 107 |
| `hw_stand_fit5v_p01` | `push2` | hardware-like (1.7 deg, 1.1 deg) | 0.1 m/s | | 75 -> 396 |
| `hw_stand_fit5v_p005` | `p01` | hardware-like | 0.05 m/s | | 732 -> ~760 |
| `hw_stand_fit5v_step` | `p01` | hardware-like | 0.1 m/s | liftoff/touchdown penalties off | 407 -> ~430 |
| `hw_stand_fit7v` | `push2` | hardware-like | 0.2 m/s | 7.4 V | 86 -> 235 |
| `hw_stand_fit7v_p01` | `fit7v` | hardware-like | 0.1 m/s | 7.4 V | 421 -> 487 |
| `hw_stand_fasthip5v` | `p01` | hardware-like | 0.1 m/s | **what-if: hips with the knees' profile** | 428 -> 805 |
| `hw_stand_fit5v_robust` | `p005` | hardware-like, tilt 2.9 deg | 0.05 m/s | backlash to 1 deg, delay to 60 ms | 579 -> ~570 |

**Lesson: match the start to the hardware.** With the slow servos even a plain stance hold survives only ~20-30 % of
the original reset, so most training episodes ended before the policy saw a shove. The robot starts from a commanded
stance (~1 deg); training from a similar start learned ~3x faster.

## 5. A non-learning balancer (simulated)

`scripts/pd_balance.py`: hold the stance and feed the IMU tilt back to the hips (pitch and pitch rate -> both hip
pitches; roll and roll rate -> hip rolls in opposite directions; optional knee terms; a slow self-trim integral). Gains
grid-searched with both signs on the fitted 5 V servos, for quiet standing and for shoves. Runs on the robot as
`policy_stand.py run --run balancer`.

* From a hardware-like start, a **plain stance hold already stands quietly in ~94 %** of randomized runs; the tuned
  feedback adds a few points. Tilt feedback through the hips adds little: with no ankles and (without the torso)
  little upper-body mass, the hips have little authority over where the weight sits on the pads, and the hip servos
  take ~200 ms to act.
* Under shoves it does worse than the retrained policies (section 6), and no gain set survived 0.1 m/s.

**Why shoves are hard without stepping (estimate).** Standing still, the largest speed the robot can absorb is about
sqrt(g/h) x (how far the centre of pressure can move). With the centre of mass ~0.30 m up, sqrt(g/h) ~ 5.7/s; the pads
allow +-15 mm fore-aft (~0.09 m/s, and only with ankle-like control of the pressure, which this robot lacks) and the
80 mm stance ~+-40 mm sideways (~0.2 m/s). Fore-aft shoves above ~0.05-0.1 m/s need a recovery step, and a step needs
fast hips.

## 6. Results on the fitted servo model (simulated, not yet tested on hardware)

`scripts/eval_fitted.py`: 24 seeded runs of 20 s, fitted servos with randomization, hardware-like start, RL policies
blended in over 1 s exactly like the robot runner. "Shoves" = a random base-velocity kick (normal, this std per axis)
every 2-4 s, ~7 per run; the percentage survived all of them. `results/eval_fitted.json`.

| controller | 5 V quiet | 5 V 0.05 m/s | 5 V 0.1 m/s | 7.4 V quiet | 7.4 V 0.05 m/s | 7.4 V 0.1 m/s |
|---|---|---|---|---|---|---|
| `hw_stand_push2` (before) | 42 % | 17 % | 0 % | 58 % | 25 % | 4 % |
| `hw_stand_fit5v_p005` | 96 % | 71 % | 8 % | 96 % | 67 % | 8 % |
| `hw_stand_fit5v_step` (free to step) | 96 % | 71 % | 4 % | 96 % | 75 % | 8 % |
| `hw_stand_fit7v` | 96 % | 67 % | 4 % | 96 % | 67 % | 8 % |
| `hw_stand_fit7v_p01` | 96 % | 54 % | 4 % | 96 % | **75 %** | 8 % |
| `hw_stand_fasthip5v` | 96 % | **79 %** | | | | |
| `hw_stand_fit5v_robust` | 96 % | 67 % | 8 % | 96 % | 67 % | |
| non-learning balancer | 88 % | 50 % | 4 % | 79 % | 42 % | 0 % |

**Initial lean** (`scripts/lean_test.py`; fitted 5 V servos, quiet, the whole robot tilted at the moment of engaging,
8 runs each, standing at 20 s):

| controller | -4 deg (back) | -2 | 0 | +2 | +4 | +6 (forward) |
|---|---|---|---|---|---|---|
| `hw_stand_push2` | 12 % | 75 % | 88 % | 0 % | 0 % | 0 % |
| `hw_stand_fit5v_p005` | 38 % | 88 % | 100 % | 62 % | 0 % | 0 % |
| `hw_stand_fit5v_robust` | 62 % | 88 % | 100 % | 62 % | 0 % | 0 % |
| `hw_stand_fasthip5v` | **75 %** | **100 %** | 100 % | 62 % | 0 % | 0 % |
| non-learning balancer | 0 % | 50 % | 88 % | 62 % | 0 % | 0 % |

Forward leans of 4 deg or more are lost for every controller, and that is statics, not control: at the stance the
centre of mass sits ~5 mm ahead of the pad centres, and each degree of forward lean moves it another ~5 mm (0.30 m up),
so past ~+2 deg it is beyond the front edge of the 30 mm pads before anything happens. **On the robot the IMU read +3
to +7 deg at the moment of engaging** in most of the 2026-11-06 trials: either it really was released already tipping
forward, or the IMU "level" is off by a few degrees. Both make a fall in the first second likely regardless of the
policy.

**What-if: hips as fast as the knees** (profile 35 rad/s^2 on the hips, everything else as fitted, 5 V):

| controller | quiet | 0.05 m/s | 0.1 m/s |
|---|---|---|---|
| `hw_stand_fasthip5v` (trained for fast hips) | 96 % | **96 %** | **62 %** |
| `hw_stand_fit5v_p005` (trained for slow hips) | 96 % | 75 % | 4 % |

(The quiet 4 % failure is one randomized robot, seed 2014, that falls at 0.8 s under every controller.)

What this says:

* **Standing quietly is solved in simulation** on the measured servos, at either supply, by retraining: 42 % -> 96 %,
  and robust to backlash and delays where the original policy was not (section 3).
* **"Slight push" resistance tops out around 0.05 m/s on the current hips** (two thirds to four fifths of runs survive
  ~7 shoves). 0.1 m/s fails ~95 % of the time for every controller, including one free to step.
* **The 7.4 V supply barely changes this.** On these servos the limits are the hip firmware's motion profile and the
  physics of a footless stance, not motor voltage. 7.4 V is still worth using (it is the servos' rating and gives
  torque margin) but it is not the fix.
* **Fast hips are the fix, together with retraining for them**: 0.1 m/s goes from ~5 % to 62 %. Fast hardware alone
  doesn't help a policy trained on slow hips.
* **Learning beats the hand-built balancer**, mostly by holding on through shoves (71-79 % vs 50 % at 0.05 m/s).
* All retrained policies sway about +-6 deg in pitch while standing (simulation): expect visible rocking, not stillness.
* Their main failure from a good start: a **forward lean of ~2 deg or more at the moment of engaging** tips them
  forward within 1 s (independent of the blend-in length), see the lean table. Training with wider start tilts
  (`hw_stand_fit5v_robust`) helps backward leans a little and cannot help forward ones.
* **Best overall: `hw_stand_fasthip5v`.** Although trained for faster hips than the robot has, on the fitted
  (slow-hip) model it is as robust as the others to backlash and delays (section 3) and the best at shoves and leans.

## 7. Running it on the robot

From `hardware/leg_p0/software`, venv active, IMU calibrated (`imu-calib`), **torso and head off** (the policies were
trained without them), IMU rigidly on the pelvis:

```bash
python policy_stand.py run --run hw_stand_fasthip5v --enable      # best on 5 V in simulation (see section 6)
python policy_stand.py run --run hw_stand_fit5v_p005 --enable     # alternative trained purely on the fitted servos
python policy_stand.py run --run hw_stand_fit7v_p01 --enable      # 7.4 V supply
python policy_stand.py run --run balancer --enable                # non-learning balancer, for comparison
```

**Check the IMU zero first.** Stand the robot in the stance pose and let it balance by itself for a moment (hands
off): `imu-check` should read pitch within ~1 deg of 0. If it reads several degrees while truly balanced, redo
`imu-calib` with the robot balanced on its pads. Then, before pressing Enter, make sure the start line's tilt is under
~1-2 deg: past ~+2 deg forward the robot is already tipping over its pads (section 6).
Enter engages (blended in over 1 s), Enter / q / space stops and holds the stance, tilt beyond `--max-tilt` cuts
torque. The runner writes servo acceleration 254 (the fitted model assumes it). Each run is logged to `stand_*.csv`.

## 8. Next steps

1. **Test on the robot**: success = standing on its own for 20 s, and riding out a light tap. If it still falls fast,
   compare joint-tracking lag in its log with the simulator's (section 1): a remaining gap points at what the model
   still misses (backlash and delay are the prime suspects, section 3).
2. **Re-run the step tests on 7.4 V** (`step-test --mode air` and `--mode stand`) and refit
   (`scripts/fit_servo.py --volt 1.0`): the hip profile at 7.4 V is assumed, not measured.
3. **Make the hips fast.** The knees (same firmware version, different EEPROM defaults) follow the acceleration register;
   the hips ignore it and ramp at ~400 deg/s^2. Candidates: an EEPROM setting in the hips (the knees' phase byte is 92,
   the hips' 12; read the Feetech manual before writing it, bits there also set direction), a firmware update, or
   bypassing the servo's position loop (PWM mode with a host-side PD at ~250 Hz). Re-run the step test after any change.
   The simulated payoff is section 6's what-if.
4. Put the torso and head back only after retraining with their mass (~350 g, high up).

## 9. Practical notes

* Training environment on a new PC: `requirements.txt` now installs `jax[cuda12]`; without the extra JAX silently
  falls back to the CPU (~100x slower).
* `policy_stand.py reset-gains` puts every servo's P gain back to 32 without moving anything (the P-gain test once
  left four hips at 96 after an interrupted run; the test now restores it even on Ctrl-C).
* Process-matching watchdogs: `scripts/queue_runner.sh` waits while any command line contains `scripts/train.py`, so a
  monitoring script containing that literal text deadlocks the queue. Match `train[.]py` instead.
