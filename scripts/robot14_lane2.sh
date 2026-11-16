#!/bin/bash
# Round 14 lane 2 (restarted 13:28 with --reset-cmd). Stage A on rs2: the slower 2.8 s step (0.22 s weight shift, 0.53 s
# swing, about k5's walking pace), random steps at 1/s after 1-2.5 s rests.
cd "$(dirname "$0")/.."
source scripts/robot14_common.sh
REC='rec_table="rs2" cmd_zero_prob=1.0 heading_cmd=false settle_gate_u=[]'
br r14c2_slow 24e6 "rec_rand=1.0 rec_rest_s=[1.0,2.5]" --reset-cmd 0.03,0.02,1.0
echo "$(date) lane done"
