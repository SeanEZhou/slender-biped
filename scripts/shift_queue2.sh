#!/bin/bash
# 2026-11-09: after shift_queue.sh: the ankle-roll walker again, with milder stepping rewards (the step-forcing set
# made it trade survival for steps, as on the 8-joint frame)
cd "$(dirname "$0")/.."
while ps -eo args | grep -v grep | grep -q "scripts/shift_queue.sh"; do sleep 30; done
EXTRA_SET='dr_floor_tc=[0.02,0.8] ref_swing_h=0.005 ref_table="b150A_shift" settle_gate_u=[0.275,0.775]' \
  scripts/b150_run.sh b150A_shift3_L shift_b150 hw_pad_board150w_legs_ankle_10dof 30e6 --rw progress=0.0 stride=20.0 ref_swing=6.0 slip=-50.0
