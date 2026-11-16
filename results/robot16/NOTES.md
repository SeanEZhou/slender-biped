# Round 16: k5's falls start at a lift-off the body isn't ready for (2026-11-15 22:00-)

> **Correction (2026-11-16, round 17's robot analysis): the sides below are swapped.** The right foot swings first
> (lift-off u 0.208, touchdown 0.396), not the left. walk_steps.py and the lift-off analysis assumed the opposite.
> - **§2:** "toward the swing foot" reads "toward the stance foot" throughout. Fatal lift-offs leaned / rolled toward
>   the stance foot. In the risk table, lift-offs rolling toward the stance foot faster than 20 deg/s were the fatal
>   ones (67 %), and those rolling toward the swing foot faster than 10 deg/s never failed (0 of 127).
> - **§4, corrected** (walk_steps.py fixed): both boards land on their inner edge. Left +3.8 deg, right -3.9 deg median
>   tonight; 48 % / 51 % beyond 5 deg. At 00:39, left +4.9, right -1.4.
> - **The attitude gate** (`settle_gate_att`) has the same swap, identically in the env and the runner. Trained and run
>   the same way, it holds a lift-off while the predicted lean toward the **stance** foot is > 3 deg, which is the
>   direction the corrected §2 data points to. Its behaviour is kept (so trained policies run as trained); only the
>   descriptions are fixed.


Back to k5's walking stability (the goal: walk quickly and stay up for days). k5 still falls after a few steps about
half the time; the operator: "a gamble on each step", and many steps "don't land squarely, which causes rocks to
build up".

## The trials (2026-11-15 19:43 and 21:37-21:51, k5_high2, `--vx 0.015 --act-lpf 0.7`, video)

[walk_steps.py](../../scripts/robot16_analysis/walk_steps.py): every step from the gait clock (touchdowns at u 0.396 /
0.896, lift-offs at 0.208 / 0.708). For each step: body roll, pitch and rates (from the attitude when the clock
starts), the landing board's tilt from the floor (encoder FK + IMU, from standing) and the gate hold. Compared with
the 00:39 batch, the best so far ([walking_clean.md §31](../../docs/walking_clean.md)).

| | 00:39 batch | 21:37-21:51 (+19:43) |
|---|---|---|
| trials | 17 (10 stopped up by the operator at ~25 s) | 13 (1 stopped up) |
| falls | 7: 4 left, 3 right | 12: **5 forward**, 4 left, 3 right |
| falls per step | 7 / 253 = 2.8 % | 12 / 177 = 6.8 % |
| standing attitude, joint angles and errors, supply volts, loop time | | the same (within 0.5 deg / 0.1 V) |

The hardware hasn't changed since 00:39 (same stance, zeros, volts).

### 1. No slow build-up: one bad step

The body's peak roll over each stance holds at ~6 deg right up to the fatal step (last 8 steps before a fall: 5.9 5.8
6.5 6.0 5.5 5.6 6.4, then **15.2**). Trials that ended standing look the same. Falls are single bad steps, not a growing
rock.

### 2. The bad step lifts off while the body isn't ready

Sideways falls: the last lift-off before a fall had the body leaning **toward the swing foot** (+3.3 deg median, p90
+10.8), where good lift-offs lean 2.6 deg the other way (over the stance foot). Risk by the body's roll rate at
lift-off (toward the swing foot +), all lift-offs of both batches:

| roll rate at lift-off | n | next stance rolls past 12 deg |
|---|---|---|
| < -10 deg/s (toward the stance foot) | 127 | 0 % |
| -10..0 | 213 | 2 % |
| 0..+10 | 86 | 10 % |
| +10..+20 | 6 | 17 % |
| > +20 | 3 | 67 % |

Video (214310): the left leg swings very high, with the knee bent nearly flat and the board flung out and twisted, while
the body rolls left over it. It lands narrow and the robot falls left, onto the foot that just swung.

Tonight's right lift-offs lean +2.0 deg toward the right foot (00:39: -0.5): the weight isn't on the left foot yet
when the right one lifts.

Forward falls (5 tonight, none at 00:39), every time:
1. A right landing kicks the body forward (pitch rate +18..+36 deg/s).
2. The next left lift-off goes ahead pitched +13..+14 deg forward.
3. It falls on its face.

### 3. Why the settle gate lets them through

k5's settle gate holds a lift-off only while the body rotates faster than 17 deg/s (roll and pitch rate together).
It never looks at the attitude or the direction. It held < 0.1 s on 90 % of steps. On the robot the gate guard
(`--gate-guard`, on by default) forced only 3 of 160 lift-offs (2 of them fatal: one rolling away, one with a foot
already up). Almost all bad lift-offs passed the gate because nothing in it was triggered.

### 4. Boards don't land squarely (the right one)

Landing board tilt from standing (world, at the clock's touchdown):

| | roll p10 / p50 / p90 | pitch p50 | tilted > 5 deg |
|---|---|---|---|
| left, tonight | -2.4 / +1.1 / +4.8 | +0.0 | 21 % |
| right, tonight | **-7.4 / -3.5 / -0.8** | +2.2 | **49 %** |
| right, 00:39 | -6.9 / -3.0 / +0.1 | -0.0 | 32 % |

The stance is asymmetric too (both batches): right hip yaw +5.7 deg, right knee 5 deg straighter than the left.

## What round 16 tries

- **An attitude gate** (`settle_gate_att`, env and runner): a lift-off also waits while the predicted lean toward the
  swing foot is > 3 deg or the predicted |pitch| > 6 deg. "Predicted" = the angle from the attitude when the clock
  starts, plus 0.2 s x the rate low-passed at 2 Hz.
  - On the robot the gate guard's "rolling away -> lift now" rule is off with it; "a foot is already up" stays.
  - On k5 alone (simulation, no training): the clock holds 0.75 s per 10 s instead of 0.16 s.
  - Runtime only: `python policy_stand.py run --run k5_high2 --enable --vx 0.015 --act-lpf 0.7 --gate-att 3,6`
- **`m1_attgate`:** k5 + the attitude gate in training (20M from k5 16M).
- **`m2_attland`:** the same + boards landing flat (`foot_tilt` -10 -> -30).

## Results (simulation)

[eval_walk.txt](eval_walk.txt): `push_eval walk`, walking 14 s at 0.015 m/s, 16 runs per direction and shove size,
% still up.
- Robot sim: the robot-like settings.
- Loose sim: the hip roll's give doubled, which reproduced k5's robot falls in round 11.

| | robot sim, no shove | robot sim, 0.1 m/s shove | robot sim overall | loose sim |
|---|---|---|---|---|
| k5 | 91 % | 81 % | 86 % | 2 % |
| k5 + the gate at run time only | 86 % | 83 % | 84 % | 0 % |
| `m1_attgate` (20M) | 92 % | 89 % | 91 % | 7 % |
| **`m2_attland`** (14M) | **95 %** | **91 %** | **93 %** | **13 %** |
| `m2b_attland` (m2 + 10M = 24M) | 83 % | 78 % | 80 % | **34 %** |

- **The gate needs training.** Added at run time to k5 it changes nothing; k5 just waits and lifts off the same way.
- **`m2` (gate + flat landings) is best** with less training: its robot-sim falls are half k5's.
- **`m2b` is mixed.** Best by far in the loose sim (34 %), worse than k5 in the robot sim (80 %), and it walks less far
  after a shove. If the robot's falls are mostly the loose-hip kind (round 11's finding), m2b may do best on it.
- **The differences are 1-2 standard errors** at 64 runs per row. The robot decides.

## To test on the robot (best first)

All with `--vx 0.015 --act-lpf 0.7 --video` as k5. The attitude gate is in the run's config, so the runner applies it
by itself and prints it at the start.

1. `python policy_stand.py run --run m2_attland --enable --vx 0.015 --act-lpf 0.7 --video`
2. `python policy_stand.py run --run m2b_attland --enable --vx 0.015 --act-lpf 0.7 --video` (robust to loose hips in
   simulation, worse on the nominal robot model)
3. `python policy_stand.py run --run m1_attgate --enable --vx 0.015 --act-lpf 0.7 --video`
4. The k5 baseline in the same session (same floor and battery), for a fair comparison:
   `python policy_stand.py run --run k5_high2 --enable --vx 0.015 --act-lpf 0.7 --video`

What to look for:
- falls per step;
- whether the robot pauses in double support before a step when it's leaning (the log's `gate_held`, `roll_pred`);
- whether the right foot lands flatter (`walk_steps.py` reports both).

5-8 trials each is enough to see a 2x change in falls per step.
