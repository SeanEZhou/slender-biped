#!/bin/bash
# Round 11, third part (20:00): p2 fell forward in 1.5 s, 6/6, and never learned. The 1 Hz low-pass on the pitch rate
# alone takes k5 from 1/6 to 6/6 falls in the fitted sim (3 Hz: 1/6): the delayed rate feedback pumps the pitch.
# p3 (p2 on the fitted hips, still 1 Hz) was stopped after 8 min. p4 = p2 with the pitch rate low-passed at 3 Hz.
cd "$(dirname "$0")/.."
source scripts/robot11_common.sh
SIDE3='obs_hold_u=[0.2083,0.3958,0.7083,0.8958] obs_hold_pitch_lpf=3.0 obs_hold_late=true'
br p4_side3_gate 20e6 "$SIDE3 $RGATE"
echo "$(date) queue3 done"
