"""The last seconds before a trial ends: per 0.5 s, 4-8 Hz gyro envelope (deg/s), roll, pitch, board height
difference L-R (mm, + = left up), and the 4-8 Hz hip-roll command (deg).   falltrace.py <csv> [seconds]"""
import sys, csv, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import gap_fit, mujoco
m, d, tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_roll", "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_roll"]
f = sys.argv[1]; S = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
r = list(csv.reader(open(f))); h = r[0]; A = np.array(r[1:], float); t = A[:, 0]
c = lambda ks: A[:, [h.index(k) for k in ks]]  # noqa: E731
g = c(["gx", "gy", "gz"]); w = np.degrees(c(["wx", "wy", "wz"])); q = c([f"q_{n}" for n in J]); cmd = np.degrees(c([f"cmd_{n}" for n in J]))
roll = np.degrees(np.arctan2(g[:, 1], -g[:, 2])); pit = np.degrees(np.arctan2(-g[:, 0], -g[:, 2]))
def bp(x):
    fr = np.fft.rfftfreq(len(x), 0.02); X = np.fft.rfft(x - x.mean(0), axis=0); X[(fr < 4) | (fr >= 8)] = 0; return np.fft.irfft(X, len(x), axis=0)
i0 = np.searchsorted(t, t[-1] - S - 4); W = bp(w[i0:, :2]); C = bp(cmd[i0:, [1, 6]])
for k in range(int(S / 0.5)):
    a = np.searchsorted(t, t[-1] - S + 0.5 * k); b = np.searchsorted(t, t[-1] - S + 0.5 * (k + 1))
    dz = []
    for i in range(a, b):
        d.qpos[:] = 0; d.qpos[2] = .3; d.qpos[3:7] = gap_fit._quat_from_g(g[i]); d.qpos[7:] = q[i]; mujoco.mj_kinematics(m, d)
        dz.append(1000 * (d.geom_xpos[tid[0], 2] - d.geom_xpos[tid[1], 2]))
    e = np.sqrt(np.mean(np.sum(W[a - i0:b - i0] ** 2, 1)))
    print(f"t-{S-0.5*k:4.1f}  shake {e:5.1f}  roll {roll[a:b].min():+5.1f}..{roll[a:b].max():+5.1f}  pitch {pit[a:b].mean():+5.1f}"
          f"  dz {min(dz):+5.1f}..{max(dz):+5.1f}  hipR cmd 4-8 {np.sqrt(np.mean(C[a-i0:b-i0]**2)):.2f}")
