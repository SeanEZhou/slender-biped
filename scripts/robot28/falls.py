"""Round 28: the robot's falls, Nov 19-20. A trial fell if, at its end, the body is tilted past 12 deg of roll or
18 deg of pitch beyond its standing pose (the user stops the run as it goes down). The fall time is when the tilt
first passed 8 deg roll / 12 deg pitch beyond the walking range for good (no return below it before the end)."""
import csv, glob, sys, json, numpy as np
D = 'hardware/leg_p0/software'
def load(f):
    r = list(csv.reader(open(f))); h = r[0]
    return h, np.array([[float(v) if v not in ('True', 'False') else float(v == 'True') for v in x] for x in r[1:] if len(x) == len(h)])
def classify(f):
    h, A = load(f); c = lambda k: A[:, h.index(k)]; t = c('t')
    G = A[:, [h.index(k) for k in ('gx', 'gy', 'gz')]]
    roll = np.degrees(np.arctan2(-G[:, 1], -G[:, 2])); pitch = np.degrees(np.arctan2(G[:, 0], -G[:, 2]))
    r0, p0 = roll[:25].mean(), pitch[:25].mean()
    u = c('phase'); moving = np.where(np.abs(np.diff(u)) > 1e-6)[0]
    out = dict(file=f.split('/')[-1], run=f.split('/')[-1][6:-20], T=float(t[-1]), steps=int(np.sum(np.abs(np.diff(np.unwrap(u)))) / np.pi))
    dr, dp = roll[-1] - r0, pitch[-1] - p0
    fell = (abs(dr) > 12) or (dp > 18) or (dp < -14)
    out['fell'] = bool(fell and len(moving) > 0)
    if out['fell']:
        bad = (np.abs(roll - r0) > 8) | (pitch - p0 > 14) | (pitch - p0 < -10)
        ok = np.where(~bad)[0]; k = ok[-1] + 1 if len(ok) else 0
        out['t_fall'] = float(t[min(k, len(t) - 1)])
        out['side'] = 'left' if dr < -8 else 'right' if dr > 8 else 'forward' if dp > 0 else 'back'
        out['roll_end'], out['pitch_end'] = round(float(dr), 1), round(float(dp), 1)
    return out
if __name__ == '__main__':
    fs = sorted(f for f in glob.glob(f'{D}/stand_*_2026111[9]_*.csv') + glob.glob(f'{D}/stand_*_20261120_*.csv') if 'frames' not in f)
    R = [classify(f) for f in fs]
    json.dump(R, open('results/robot28/falls.json', 'w'), indent=1)
    for r in R:
        print(f"{r['file'][6:-4]:28s} T {r['T']:5.1f} steps {r['steps']:3d} " + (f"FELL {r['side']:7s} at {r['t_fall']:5.1f}s (end roll {r['roll_end']:+.0f} pitch {r['pitch_end']:+.0f})" if r['fell'] else "-"))
    print(f"{sum(r['fell'] for r in R)} falls in {len(R)} trials")
