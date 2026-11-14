# Round 6-7 analysis scripts (2026-11-13)

Working scripts behind docs/walking_clean.md §20-22 and measuring.md E8, kept as they were run. Paths are relative
to the repository root: run them from there, `.venv/bin/python scripts/robot7_analysis/<script>`.

- `wig.py <ex_wiggle_*.csv>`: per-frequency joint following and gyro per degree commanded (E8).
- `fitsim.py [k=v ...]`, `grid.py <pattern> '<json grid>'`, `phase.py '<json list>'`: the simulator's wiggle response
  against the robot's; how the ring was fitted.
- `replay.py`, `ablate.py`, `jac.py`, `lpfcheck.py`: replay a robot log's observations through its policy (exact),
  find which input carries the 4-8 Hz chatter, the policy's gyro gain, and the action low-pass a trial used.
- `plant.py`, `onset.py`, `trial2.py`, `posture.py`, `lifts.py`: shake, onset, steps, lean and posture, robot vs sim.
- `histcheck.py`: the robot runner's history observation against the evaluator's.
