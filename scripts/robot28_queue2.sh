#!/bin/bash
# round 28 queue 2 (13:45): z8_b3 (30 % bank, half of it states at / after the fall onset) lost normal walking (48-67 %
# up) and didn't improve the held-out falls. Milder banks instead of z8_b5mir:
#   robot28pre  the normal robot states + the bad states up to the onset only (~19 %)
#   robot28nor  the normal robot states only
cd "$(dirname "$0")/.."; source scripts/robot28_common.sh
late() { [ $(date +%H%M) -gt 1615 ] && { echo "$(date) past 16:15, not starting $1"; return 0; }; return 1; }
until [ -f runs/z8_b3ng/done.json ]; do sleep 20; done
pkill -f "^/bin/bash scripts/robot28_queue.sh"; sleep 2
ev28 z8_b3ng
late z8_ctlng || L28 z8_ctlng 20e6 "$A2 $NG"         land_v=-100.0 $LATE   # control: no bank, gate-free
late z8_pre3ng || L28 z8_pre3ng 20e6 "$A2 reset_bank=\"robot28pre\" reset_bank_prob=0.3 $NG" land_v=-100.0 $LATE
# 15:00: the control z8_ctlng drifted too (90/69/50 %, loose 0): a lower learning rate for the last one
late z8_preclp || L28 z8_preclp 20e6 "$A2 reset_bank=\"robot28preclp\" reset_bank_prob=0.3 $NG" land_v=-100.0 $LATE --lr 1e-4   # 15:25: bank velocities capped at 0.08 m/s (19 % of the fits sat at the 0.3 m/s grid edge)
# 16:02: z8_ctllr dropped: the remaining GPU time goes to reproducing all the falls
for r in z7_anc2 z0_y2c z6_e z7_rgq z7_var z5_shv10 z5_shv2 k5_high2 h4s2_t10 h4x2_t10 h9s_t09 g0g_gate g7g_t11; do
  [ -f results/robot28/repro_${r}_logged.json ] && [ results/robot28/repro_${r}_logged.json -nt results/robot28/holdout.json ] && continue
  [ $(date +%H%M) -gt 1712 ] && break
  scripts/gpu_python.sh scripts/robot28/repro.py $r --controls 2 2>&1 | grep -vi "$FLT" >> results/robot28/repro_all.log
done
echo "$(date) queue2 done"
