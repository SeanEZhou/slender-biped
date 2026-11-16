#!/bin/bash
# Round 14 lane 3 (13:58, replaces the fast-step lane): at 12M both stage-A runs kept their feet down through most
# scheduled swings (ref_swing paid ~2-10 %); the slow step (rs2) lifted 4x more. From r14c2_slow at 12M, the swing pay
# raised: ref_swing 6 -> 20, paid fully at 1.2 cm instead of 2.5 (ref_swing_h), the lift shortfall clear -2 -> -5.
cd "$(dirname "$0")/.."
source scripts/robot14_common.sh
REC='rec_table="rs2" cmd_zero_prob=1.0 heading_cmd=false settle_gate_u=[]'
INIT=runs/r14c2_slow/params_0012M.pkl
br r14e_swing 18e6 "rec_rand=1.0 rec_rest_s=[1.0,2.5] ref_swing_h=0.012" ref_swing=20.0 clear=-5.0
echo "$(date) lane done"
