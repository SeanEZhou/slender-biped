#!/bin/bash
# round 9 branches from g1_t10l 40M, one at a time (one run takes 3.4 GB of the 7.6 GB and the whole GPU); steps per
# branch from runs/robot9_steps (default 24e6), read at each start
cd "$(dirname "$0")/.."; . scripts/robot9_common.sh
until [ -f $BASE_INIT ] && ! ps -eo args | grep -q "^[^ ]*python[^ ]* scripts/train.py --name g1_t10l"; do sleep 30; done
br d1b_ignore ""                                  # no ring: ignore the problem, filter at run time only (d1_ignore: old pay)
br d7_vib "$R05 gyro_vib=[0.35,3.5,7.0]"          # fake 3.5-7 Hz gyro bursts up to 20 deg/s: answering them only costs
br d5_hf "$R05 act_lpf=0.6" act_hf=-0.5           # filter trained in + penalty on the raw output above 2 Hz
br d6_glcp "$R05" --lcp 0.05 --lcp-idx 3:6        # gradient penalty on the gyro inputs only
br d2b_ring05 "$R05"                              # half the round-7 ring (d2_ring05: old pay, stopped walking)
br d3_ring10 "$R10"                               # the round-7 ring
br d9_blind "$R05 gyro_obs_scale=0.0"             # no gyro: the loop through the gyro is gone (gravity and joints remain)
br d8_glpf "$R05 gyro_lpf=2.5"                    # 2.5 Hz gyro low-pass trained in (runs with it on the robot)
br d10_land "$R05" land_v=-200.0                  # soft touchdowns: the shake floor comes from the landings
br d4_ring15 "$R15"                               # 1.5x: past the robot's
br d11_combo "$R05 gyro_vib=[0.35,3.5,7.0] act_lpf=0.6" land_v=-200.0 act_hf=-0.5
