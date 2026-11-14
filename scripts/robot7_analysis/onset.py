import sys, csv, os, numpy as np
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
def bp(x, dt, a=3.5, b=8):
    fr = np.fft.rfftfreq(len(x), dt); X = np.fft.rfft(x - x.mean(0), axis=0); X[(fr < a) | (fr >= b)] = 0
    return np.fft.irfft(X, len(x), axis=0)
for f in sys.argv[1:]:
    rows = list(csv.reader(open(f))); h = rows[0]; A = np.array(rows[1:], float); t = A[:, 0]; dt = np.median(np.diff(t))
    c = lambda ks: A[:, [h.index(k) for k in ks]]
    w = np.degrees(c(["wx","wy","wz"])); cmd = np.degrees(c([f"cmd_{n}" for n in J])); act = c([f"a_{n}" for n in J]); q = np.degrees(c([f"q_{n}" for n in J]))
    g = c(["gx","gy","gz"]); roll = np.degrees(np.arctan2(g[:,1], -g[:,2])); pitch = np.degrees(np.arctan2(-g[:,0], -g[:,2]))
    eng = np.argmax(np.any(act != 0, 1))
    wb = bp(w, dt); cb = bp(cmd, dt); qb = bp(q, dt)
    print(f"\n{os.path.basename(f)[6:-4]}  policy on at {t[eng]:.1f} s, end {t[-1]:.1f}")
    print("   t   roll pitch | gyro 3.5-8Hz rms r/p/y deg/s | cmd 3.5-8Hz rms deg (yawL yawR rollL rollR pitL pitR kneeL kneeR ankL ankR) | enc same")
    n = int(0.5 / dt)
    for i in range(max(eng - 2*n, 0), len(t) - n, n):
        s = slice(i, i + n)
        r = lambda x: np.sqrt(np.mean(x[s] ** 2, 0))
        cr = r(cb); qr = r(qb)
        order = [0,5,1,6,2,7,3,8,4,9]
        print(f"{t[i]:5.1f} {roll[i]:+5.1f} {pitch[i]:+5.1f} | " + " ".join(f"{x:5.1f}" for x in r(wb)) + " | " + " ".join(f"{cr[k]:4.2f}" for k in order) + " | " + " ".join(f"{qr[k]:4.2f}" for k in order))
