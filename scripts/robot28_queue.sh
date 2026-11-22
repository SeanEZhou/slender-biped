#!/bin/bash
# round 28 queue (GPU budget 11:23-17:23; nothing new after 16:10)
cd "$(dirname "$0")/.."; source scripts/robot28_common.sh
late() { [ $(date +%H%M) -gt 1610 ] && { echo "$(date) past 16:10, not starting $1"; return 0; }; return 1; }
while [ ! -f models/bank/robot28.npz ]; do sleep 20; done
echo "$(date) bank ready"
# training needs the RAM: stop the all-falls reproduction (finished runs keep their json); the rest runs at the end
pkill -f "^/bin/bash scripts/robot28/repro_all.sh" ; pkill -f "^scripts/../.venv/bin/python scripts/robot28/repro.py" ; sleep 5
# baselines on the held-out falls (the policies the robot ran, gate-free)
evhold z7_anc2:params_0012M.pkl; evhold z0_y2c:params_0020M.pkl
late z8_b3    || L28 z8_b3    20e6 "$A2 $BANK"       land_v=-100.0 $LATE   # bank 30 %, gated (as z7_anc2 trained)
late z8_b3ng  || L28 z8_b3ng  20e6 "$A2 $BANK $NG"   land_v=-100.0 $LATE   # same, gate-free training
late z8_ctlng || L28 z8_ctlng 20e6 "$A2 $NG"         land_v=-100.0 $LATE   # control: no bank, gate-free
late z8_b5mir || L28 z8_b5mir 20e6 "$A2 reset_bank=\"robot28\" reset_bank_prob=0.5 mirror_aug=true $NG" land_v=-100.0 $LATE
# the reproduction of the remaining runs' falls
for r in z7_anc2 z0_y2c z6_e z7_rgq z7_var z5_shv10 z5_shv2 k5_high2 h4s2_t10 h4x2_t10 h9s_t09 g0g_gate g7g_t11; do
  [ -f results/robot28/repro_${r}_logged.json ] && [ results/robot28/repro_${r}_logged.json -nt results/robot28/holdout.json ] && continue
  scripts/gpu_python.sh scripts/robot28/repro.py $r --controls 2 2>&1 | grep -vi "$FLT" >> results/robot28/repro_all.log
done
echo "$(date) queue done"
