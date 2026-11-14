#!/bin/bash
# round 9 base: h4_t05 -> the c_T10L gait on h4's own (round-6, no ring) simulator, 40M
cd "$(dirname "$0")/.."; . scripts/robot9_common.sh
echo "$(date) start g1_t10l"
run g1_t10l $F 40e6 "$W $LEAN $SER" --init runs/h4_t05/params.pkl --rw $STEP $C3M tilt_lim=-0.05
echo "$(date) end g1_t10l rc=$?"
