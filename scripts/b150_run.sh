#!/bin/bash
# Walking / skiing on the F_board150w pads (2026-11-08): the pad screening recipe's servo, delay and rug randomization.
#   [EXTRA_SET="k=json ..."] scripts/b150_run.sh <name> <task> <model> <steps> [extra train.py args...]
cd "$(dirname "$0")/.."
name=$1; task=$2; model=$3; steps=$4; shift 4
scripts/gpu_python.sh scripts/train.py --name $name --model $model --task $task --steps $steps --dr --push 0.0 \
  --action-scale 0.25 \
  --set 'prof_acc=[7.0,35.0,35.0]' 'dr_delay_steps=[0,1]' 'dr_latency_sub=[2,10]' 'dr_prof_acc=[0.6,1.5]' \
        'dr_volt=[0.6,1.05]' 'dr_friction=[0.4,1.2]' 'dr_floor_tc=[0.02,0.3]' 'dr_floor_damp=[0.7,2.0]' $EXTRA_SET \
  "$@" > runs/$name.log 2>&1 < /dev/null
