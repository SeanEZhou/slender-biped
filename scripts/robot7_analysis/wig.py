import sys, csv, json, numpy as np
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
PAT = {"roll": np.array([0,-1,0,0,1,0,1,0,0,-1.0]), "pitch": np.array([0,0,1,0,0,0,0,1,0,0.]), "knees": np.array([0,0,0,1,0,0,0,0,1,0.]),
       "yaw": np.array([1,0,0,0,0,1,0,0,0,0.]), "squat": np.array([0,0,1,1,0,0,0,1,1,0.])}
def lockin(t, x, f):
    M = np.column_stack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t), np.ones_like(t), t - t.mean()])
    c = np.linalg.lstsq(M, x, rcond=None)[0]; return c[0] + 1j*c[1]      # complex amplitude vs sin
def load(f):
    rows = list(csv.reader(open(f))); h = rows[0]; st = np.array([r[h.index("stage")] for r in rows[1:]])
    A = np.array([[float(v) for v in r[:h.index("stage")]] for r in rows[1:]]); meta = json.load(open(f[:-4] + ".json"))
    return h, A, st, meta
def analyse(f, quiet=False):
    h, A, st, meta = load(f); t = A[:, 0]; pat = meta["pattern"]; p = PAT[pat]
    c = lambda ks: A[:, [h.index(k) for k in ks]]
    w = np.degrees(c(["wx","wy","wz"])); q = np.degrees(c([f"q_{n}" for n in J])); cmd = np.degrees(c([f"cmd_{n}" for n in J]))
    out = []
    if not quiet: print(f"\n{f.split('/')[-1]}  {pat} amp {meta['amp_deg']}  stop: {meta['stop']}")
    for s in dict.fromkeys(st):
        if not s.endswith("Hz"): continue
        fr = float(s[:-2]); m = st == s; tm = t[m]; m &= (t > tm[0] + 0.4) & (t < tm[-1] - 0.3)
        if m.sum() < 40: continue
        uc = lockin(t[m], cmd[m] @ p / (p @ p), fr); uq = lockin(t[m], q[m] @ p / (p @ p), fr)
        W = np.array([lockin(t[m], w[m, k], fr) for k in range(3)])
        k = int(np.argmax(np.abs(W)))
        rel = W / uc                                         # deg/s per deg commanded, phase vs command
        out.append((fr, np.abs(uc), np.abs(uq / uc), np.angle(uq / uc, 1), np.abs(rel), np.angle(rel, 1)))
        if not quiet: print(f"  {fr:4.1f} Hz cmd {abs(uc):4.2f} deg  joint/cmd {abs(uq/uc):4.2f} @ {np.angle(uq/uc,1):+5.0f}  |  gyro per deg cmd r/p/y "
              + " ".join(f"{abs(x):5.1f}@{np.angle(x,1):+4.0f}" for x in rel) + f"   gyro amp {abs(W[k]):5.1f} deg/s ({'rpy'[k]})")
    return out
if __name__ == "__main__":
    for f in sys.argv[1:]: analyse(f)
