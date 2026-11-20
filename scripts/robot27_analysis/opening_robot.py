"""Round 27: the robot's opening. Per trial, the first three swings vs the later ones: at lift-off the body roll / roll
rate (deg, deg/s, + = right), the hip rolls (deg, + = outward) measured and commanded, the table's hip rolls, and the
swing foot's landing (dx ahead of the stance foot / dy apart, mm)."""
import sys, csv, numpy as np
sys.path.insert(0, 'scripts/robot25_analysis'); sys.path.insert(0, 'scripts/robot18_analysis'); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
from steps25 import feet, s
from slender_biped.ref_gait import lookup
J = list(s.J); iLr, iRr = J.index('L_hip_roll'), J.index('R_hip_roll')
for f in sys.argv[1:]:
    r = list(csv.reader(open(f))); h = r[0]
    A = np.array([[float(v) if v not in ('True', 'False') else float(v == 'True') for v in x] for x in r[1:] if len(x) == len(h)])
    run = f.split('/')[-1][6:-20]; tab, z, ev = s.table(run); st = z['stance'][-1]; n = len(st)
    c = lambda k: A[:, h.index(k)]; u = c('phase') / 2 / np.pi; G = A[:, [h.index(k) for k in ('gx', 'gy', 'gz')]]
    Q = A[:, [h.index('q_' + j) for j in J]]; C = A[:, [h.index('cmd_' + j) for j in J]]
    roll = np.degrees(np.arctan2(-G[:, 1], -G[:, 2])); wx = np.degrees(c('wx'))
    vx = c('vx') if 'vx' in h else np.full(len(u), np.nan)
    k0 = np.where(np.abs(np.diff(u)) > 1e-6)[0][0]; r0 = roll[k0 - 25:k0].mean()
    out = []
    for i in range(k0, len(u) - 8):
        p0 = int((u[i] % 1) * n) % n; p1 = int((u[i + 1] % 1) * n) % n
        for sd in (0, 1):
            if st[p0][sd] >= .5 and st[p1][sd] < .5:
                nt = [j for j in range(i + 1, min(i + 60, len(u) - 1)) if st[int((u[j] % 1) * n) % n][sd] >= .5]
                if not nt: continue
                j = min(nt[0] + 6, len(u) - 1); L, R = feet(Q[j], G[j]); sw, sdd = (L, R) if sd == 0 else (R, L)
                qr = np.asarray(lookup(z, max(vx[i], 0.0) if np.isfinite(vx[i]) else 0.02, u[i] * 2 * np.pi)[0])
                out.append(f"{'LR'[sd]} t{(i - k0) * 0.02:5.2f} vx {vx[i]:.3f} roll {roll[i] - r0:+5.1f} {wx[i]:+6.1f}/s  hiproll L/R q {np.degrees(Q[i, iLr]):+5.1f}/{np.degrees(Q[i, iRr]):+5.1f} "
                           f"cmd {np.degrees(C[i, iLr]):+5.1f}/{np.degrees(C[i, iRr]):+5.1f} table {np.degrees(qr[iLr]):+5.1f}/{np.degrees(qr[iRr]):+5.1f} | land {(sw[0] - sdd[0]) * 1000:+4.0f}/{abs(L[1] - R[1]) * 1000:3.0f}")
    print(f.split('stand_')[1][:-4]); print('   ' + '\n   '.join(out[:4])); print('   ... later: ' + out[8] if len(out) > 8 else '')
