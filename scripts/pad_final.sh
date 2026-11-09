#!/bin/bash
# Final rug policy for one pad design: the screening run's settings plus the robot's IMU in the loop (firmware
# complementary filter with the new settings: +-1000 dps gyro, accelerometer gated at 0.15 g), longer, from the
# screening checkpoint.
#   scripts/pad_final.sh <design> <steps>
cd "$(dirname "$0")/.."
d=$1; steps=${2:-100e6}
init=$(ls runs/pad_${d}/params_*M.pkl | sort | tail -1)
scripts/gpu_python.sh scripts/train.py --name rug_${d} --model hw_pad_${d}_8dof --task stand_quiet2 --steps $steps --dr \
  --push 0.12 --init $init --rw liftoff=0 touchdown=0 both_feet=0.1 \
  --set 'prof_acc=[7.0,35.0,35.0]' 'dr_delay_steps=[0,1]' 'dr_latency_sub=[2,10]' 'dr_prof_acc=[0.6,1.5]' \
        'dr_volt=[0.6,1.05]' reset_joint_noise=0.05 reset_tilt=0.05 reset_hard_prob=0.5 'reset_hard_tilt=[0.05,0.16]' \
        reset_angvel=0.3 'dr_friction=[0.4,1.2]' 'dr_floor_tc=[0.02,0.3]' 'dr_floor_damp=[0.7,2.0]' \
        imu_filter=true imu_gate=0.15 gyro_range=17.45 \
  --lr 1e-4 --entropy 0.005 > runs/rug_${d}.log 2>&1 < /dev/null
