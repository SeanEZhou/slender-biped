"""Twitch bouts on the robot (round 8 trials, 2026-11-13 22:16-22:30). Per trial: the 4-8 Hz gyro (roll / pitch, deg/s)
and command (hip roll, hip pitch, knee, deg) while both boards are down vs while one swings, the share of time in a
bout (4-8 Hz gyro envelope > 25 deg/s), the loop ratio (4-8 Hz command per 4-8 Hz gyro, deg per deg/s), the
largest roll, and how the trial ended.

    .venv/bin/python scripts/robot8_analysis/twitch.py <stand_*.csv> ...
"""
import sys, csv, os, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import gap_fit, mujoco
m, d, tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_roll", "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_roll"]

def bp(x, lo=4.0, hi=8.0):
    fr = np.fft.rfftfreq(len(x), 0.02); X = np.fft.rfft(x - x.mean(0), axis=0); X[(fr < lo) | (fr >= hi)] = 0
    return np.fft.irfft(X, len(x), axis=0)

def env(x, n=25):   # rms over a 0.5 s window
    k = np.ones(n) / n; return np.sqrt(np.convolve(x ** 2, k, "same"))

for f in sys.argv[1:]:
    r = list(csv.reader(open(f))); h = r[0]; A = np.array(r[1:], float); t = A[:, 0]
    c = lambda ks: A[:, [h.index(k) for k in ks]]  # noqa: E731
    g = c(["gx", "gy", "gz"]); w = np.degrees(c(["wx", "wy", "wz"])); q = c([f"q_{n}" for n in J]); a = c([f"a_{n}" for n in J])
    cmd = np.degrees(c([f"cmd_{n}" for n in J]))
    eng = int(np.argmax(np.any(a != 0, 1)))
    s = slice(eng + 25, len(t) - 25)
    roll = np.degrees(np.arctan2(g[:, 1], -g[:, 2])); pit = np.degrees(np.arctan2(-g[:, 0], -g[:, 2]))
    dz = []
    for i in range(len(t)):
        d.qpos[:] = 0; d.qpos[2] = .3; d.qpos[3:7] = gap_fit._quat_from_g(g[i]); d.qpos[7:] = q[i]; mujoco.mj_kinematics(m, d)
        dz.append(1000 * (d.geom_xpos[tid[0], 2] - d.geom_xpos[tid[1], 2]))
    dz = np.array(dz)[s]
    W = bp(w[s][:, :2]); C = bp(cmd[s])
    ew = env(np.sqrt(np.sum(W ** 2, 1)))
    ds = np.abs(dz) < 3; sw = np.abs(dz) > 5
    rms = lambda x, m_: np.sqrt(np.mean(x[m_] ** 2, 0)) if m_.sum() > 20 else np.full(x.shape[1:], np.nan)  # noqa: E731
    cr = lambda m_: np.mean([rms(C[:, [1, 6]], m_).mean(), rms(C[:, [2, 7]], m_).mean(), rms(C[:, [3, 8]], m_).mean()])  # noqa: E731
    wg = rms(W, ds); wsw = rms(W, sw)
    loop = cr(np.ones(len(W), bool)) / max(np.sqrt(np.mean(np.sum(W ** 2, 1))), 1e-6)
    bout = ew > 25
    T = t[-1] - t[eng]
    tb = t[s][bout]
    first = f"{tb[0]-t[eng]:5.1f}" if len(tb) else "   - "
    print(f"{os.path.basename(f)[6:-4]:34s} {T:6.1f} s  gyro 4-8 Hz r/p both down {wg[0]:4.1f}/{wg[1]:4.1f} swinging {wsw[0]:4.1f}/{wsw[1]:4.1f}"
          f"  cmd 4-8 (hipR hipP knee) {rms(C[:, [1, 6]], np.ones(len(C), bool)).mean():.2f} {rms(C[:, [2, 7]], np.ones(len(C), bool)).mean():.2f} {rms(C[:, [3, 8]], np.ones(len(C), bool)).mean():.2f} deg"
          f"  loop {loop:.3f}  in bout {100*bout.mean():3.0f}% first {first}s  |roll| max {np.abs(roll[s]).max():4.1f}  end roll {roll[-1]:+4.0f} pitch {pit[-1]:+4.0f}")
