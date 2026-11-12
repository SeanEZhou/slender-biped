# Robot round 4 plan (overnight 2026-11-12)

Goal: a policy steadier than c3m_T12 with longer steps, under the measured limits (sideways tip 6-7 deg, feet slip
inward from 4-5 deg, the stance narrows each sway).

Sim calibration (done): scripts/tilt_test_sim.py; hip_roll hidden give k=4 N*m/rad gives tip ~8.5, slip ~5.7, gap
81 -> 51 mm when let go. Screening at k=4 (scripts/robot4_eval.py): c3m and c3f 0/6 falls, h13 and h9 6/6, as on
the robot. Gate for any new policy: robot4_eval at k=4 and 5, vx 0.015-0.03.

Training env: mjx_env hidden_flex + dr_flex (k 3-6), roll_lim reward (pelvis roll past roll_lim rad, deg^2).

Queue 1 ran r1_flex, r2_lim; then queue 3 (scripts/robot4_queue3.sh): r3_long, r5_mild, p1_w15, r4_t15, p2_w25.
Next after queue 1: pick the best of r2-r4; try roll_lim weight -3, wider ref (c_T12w3), or more steps (fine-tune
the best another 20 M). Then the outer half-circle pad study (sim: foot geoms extended outward).
Queue 4 (after r3 stepped in place): s1_fwd, s2_long, s3_w15, s4_mild, s5_w25 (progress=1, stride=40, init r1_flex).
Queue 5 (after 4): s6_wide (s2 + give k 2.4-6, cmd 0.015-0.04), s7_narrow (s6 + narrow only).
Queue 6: s8_push (from s7: progress 2, stride 60, narrow), s9_t15 (from s7 on c_T15).
Queue 7 (after 6): s10_robust (s7 + k 2-6 + IMU tilt dr), s11_pushn (s8 + narrow x3 + k 2-6).
