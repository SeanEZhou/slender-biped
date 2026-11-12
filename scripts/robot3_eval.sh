#!/bin/bash
# 2026-11-11 robot round 3 evaluation, robot-like: faster servos (as measured on the robot), robot-like speed noise,
# hip rolls 2.5x stiffer (kp_hip_roll; the robot's standing hip roll sags 2.5x less than the old model's and its pelvis
# lifts the other foot 44-59 mm inside the standing board, the old model's 21-31), 5 V and 7.4 V, 6 x 30 s; then 5 V
# with the old soft hip model for reference.   scripts/robot3_eval.sh <run> [--params P] [--vx V]
cd "$(dirname "$0")/.."
P="JAX_PLATFORMS=cpu .venv/bin/python scripts/robot_probe.py"
JAX_PLATFORMS=cpu .venv/bin/python scripts/robot_probe.py "$@" --n 6 --seconds 30 --volt 0.68,1.0 --delay 0 \
  --set "prof_acc=(25.0,60.0,60.0)" "noise_qd=0.25" "dr_kp=(0.75,1.25)" "kp_hip_roll=2.5" 2>&1 | grep -v -i warp
JAX_PLATFORMS=cpu .venv/bin/python scripts/robot_probe.py "$@" --n 6 --seconds 30 --volt 0.68 --delay 0 \
  --set "prof_acc=(25.0,60.0,60.0)" "noise_qd=0.25" "dr_kp=(0.75,1.25)" "kp_hip_roll=1.0" 2>&1 | grep -v -i warp | tail -1 | sed 's/^  volt/  SOFT-HIP volt/'
