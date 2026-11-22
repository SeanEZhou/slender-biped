"""Round 27 robot trials: the left 'limp'. Per side, over the steady steps (5th on, before any fall): the swing foot's
landing dx ahead of the stance foot / dy apart (mm, 0.12 s after the scheduled touchdown), the actual swing time (FK
foot height above the stance foot > 4 mm) vs the table's, the body roll at touchdown (deg, + = right, vs standing),
and at the scheduled touchdown the swing leg's hip pitch / knee: commanded vs measured vs the table (deg)."""
import sys, csv, numpy as np
sys.path.insert(0, 'scripts/robot25_analysis'); sys.path.insert(0, 'scripts/robot18_analysis'); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
from steps25 import feet, s
from slender_biped.ref_gait import lookup
J = list(s.J)
def load(f):
    r = list(csv.reader(open(f))); h = r[0]
    A = np.array([[float(v) for v in x] for x in r[1:] if len(x) == len(h)]); return h, A
def analyse(f):
    h, A = load(f); run = f.split('/')[-1][6:-20]; tab, z, ev = s.table(run); st = z['stance'][-1]; n = len(st)
    c = lambda k: A[:, h.index(k)]; u = c('phase') / 2 / np.pi; G = A[:, [h.index(k) for k in ('gx', 'gy', 'gz')]]
    Q = A[:, [h.index('q_' + j) for j in J]]; C = A[:, [h.index('cmd_' + j) for j in J]]
    vx = c('vx_cmd') if 'vx_cmd' in h else np.full(len(u), 0.02)
    roll = np.degrees(np.arctan2(-G[:, 1], -G[:, 2]))
    k0 = np.where(np.abs(np.diff(u)) > 1e-6)[0][0]; r0 = roll[k0 - 25:k0].mean()
    bad = np.where((np.abs(roll - r0) > 15) & (np.arange(len(u)) > k0))[0]; b = bad[0] if len(bad) else len(u) - 1
    FK = [feet(Q[i], G[i]) for i in range(len(u))]
    out = []
    for i in range(k0, b - 8):
        p0 = int((u[i] % 1) * n) % n; p1 = int((u[i + 1] % 1) * n) % n
        for sd in (0, 1):
            if st[p0][sd] >= .5 and st[p1][sd] < .5:          # scheduled lift-off
                nt = [j for j in range(i + 1, min(i + 80, b - 7)) if st[int((u[j] % 1) * n) % n][sd] >= .5]
                if not nt: continue
                td = nt[0]; j = td + 6
                hgt = np.array([(FK[k][sd][2] - FK[k][1 - sd][2]) * 1000 for k in range(i, min(td + 30, b))])
                air = np.where(hgt > 4)[0]
                t_air = (air[-1] - air[0] + 1) * 0.02 if len(air) else 0.0
                L, R = FK[j]; sw, sdd = (L, R) if sd == 0 else (R, L)
                qr = np.degrees(np.asarray(lookup(z, max(vx[td], 0.0), u[td] * 2 * np.pi)[0]))
                pre = 'LR'[sd] + '_'
                ih, ik = J.index(pre + 'hip_pitch'), J.index(pre + 'knee')
                out.append(dict(side='LR'[sd], dx=(sw[0] - sdd[0]) * 1000, dy=abs(L[1] - R[1]) * 1000, t_air=t_air,
                                t_sched=(td - i) * 0.02, hmax=hgt.max(), roll_td=roll[td] - r0,
                                roll_min=(roll[i:j] - r0).min(), roll_max=(roll[i:j] - r0).max(),
                                hp=(np.degrees(C[td, ih]), np.degrees(Q[td, ih]), qr[ih]),
                                kn=(np.degrees(C[td, ik]), np.degrees(Q[td, ik]), qr[ik])))
    return run, out
if __name__ == '__main__':
    agg = {}
    for f in sys.argv[1:]:
        run, o = analyse(f); o = o[4:]
        agg.setdefault(run, []).extend(o)
    for run, o in agg.items():
        print(f"{run}  ({len(o)} steady steps)")
        for sd in 'LR':
            x = [e for e in o if e['side'] == sd]
            if not x: continue
            m = lambda k: np.median([e[k] for e in x])
            hp = np.median([e['hp'] for e in x], 0); kn = np.median([e['kn'] for e in x], 0)
            print(f"  {sd}: dx {m('dx'):+4.0f} (p10 {np.percentile([e['dx'] for e in x], 10):+4.0f}) dy {m('dy'):3.0f} | air {m('t_air'):.2f}s of {m('t_sched'):.2f} "
                  f"peak {m('hmax'):4.1f}mm | roll at td {m('roll_td'):+5.1f} range {m('roll_min'):+5.1f}..{m('roll_max'):+5.1f} "
                  f"| hip pitch cmd/q/table {hp[0]:+5.1f}/{hp[1]:+5.1f}/{hp[2]:+5.1f} knee {kn[0]:+5.1f}/{kn[1]:+5.1f}/{kn[2]:+5.1f}")
