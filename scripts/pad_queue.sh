#!/bin/bash
# Screening queue: one stand/recovery run per foot-pad design on rug-soft floors (scripts/make_pads.py models).
#   scripts/pad_queue.sh <steps> <design> [<design> ...]
cd "$(dirname "$0")/.."
steps=$1; shift
for d in "$@"; do
  name=pad_${d}
  [ -f runs/$name/done.json ] && continue
  scripts/gpu_python.sh scripts/train.py --name $name --model hw_pad_${d}_8dof --task stand_quiet2 --steps $steps --dr \
    --push 0.12 --init runs/hw_recover_soft/params_0121M.pkl --rw liftoff=0 touchdown=0 both_feet=0.1 \
    --set 'prof_acc=[7.0,35.0,35.0]' 'dr_delay_steps=[0,1]' 'dr_latency_sub=[2,10]' 'dr_prof_acc=[0.6,1.5]' \
          'dr_volt=[0.6,1.05]' reset_joint_noise=0.05 reset_tilt=0.05 reset_hard_prob=0.5 'reset_hard_tilt=[0.05,0.16]' \
          reset_angvel=0.3 'dr_friction=[0.4,1.2]' 'dr_floor_tc=[0.02,0.3]' 'dr_floor_damp=[0.7,2.0]' \
    --lr 1e-4 --entropy 0.005 > runs/$name.log 2>&1 < /dev/null
done
