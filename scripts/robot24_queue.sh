#!/bin/bash
# round 24 stage 1: z0 fine-tuned onto the four stride tables, gate off, z0's recipe otherwise (falls cost 20).
cd "$(dirname "$0")/.."; source scripts/robot24_common.sh
L s4h_z0   0.5   20e6 "$LANDV $S4H $NG" land_v=-100.0 $LATE
L s3h_z0   0.4   20e6 "$LANDV $S3H $NG" land_v=-100.0 $LATE
L s4_z0    0.5   20e6 "$LANDV $S4 $NG" land_v=-100.0 $LATE
L s3_z0    0.4   20e6 "$LANDV $S3 $NG" land_v=-100.0 $LATE
echo "$(date) queue done"
