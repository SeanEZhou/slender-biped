"""Round 25 (2026-11-19): the opening steps in the GPU 'sim robot' (gait_eval's robot settings), started as the robot
runner starts (policy_stand --ready-act): the clock held at 0 with a zero command for --hold s, then the command.

Per walker, from the encoders (+ hidden give) and the body attitude through the rigid model (as the robot's logs are
read): each of the first steps' landing foot ahead of the stance foot (dx) and the foot spacing (dy), 0.12 s after the
scheduled touchdown; the body roll excursion over the first 3 s after the release; the hip rolls at the end of the hold.
Robot reference (z0_y2c, 2026-11-19 11:48-12:01, scratchpad steps25.py): first R +12..+22 / 135-140, first L +6..+12 /
106-124, steady R +38 / L +15 (mm ahead), spacing ~131.

    scripts/gpu_python.sh scripts/robot25_analysis/opening.py <run> [--vx 0.025] [--hold 2.0] [--set k=json ...]
"""
import argparse, functools, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "robot18_analysis"))
import jax, jax.numpy as jp, numpy as np, mujoco
from brax.envs.wrappers import training as bw
from push_eval import ROBOT, load_policy
from slender_biped.mjx_env import SlenderBipedEnv, randomization_fn
from slender_biped.runcfg import env_config
import swing_clear as sc

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--vx", type=float, default=0.025); ap.add_argument("--hold", type=float, default=2.0)
ap.add_argument("--n", type=int, default=32); ap.add_argument("--T", type=float, default=8.0); ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
W, p, d = load_policy(a.run)
over = dict(step_err=(), reset_stagger=0.0, push_dv=(0.0, 0.0), push_interval_s=(1000.0, 1000.0), push_vel=0.0, act_lpf=0.7, cmd_resample_s=(1000.0, 1000.0),
            reset_hard_prob=0.0, reset_joint_noise=0.02, reset_tilt=0.02, cmd_vx=(a.vx, a.vx), cmd_zero_prob=0.0,
            cmd_wz=(0.0, 0.0), sup=False, start_hold_s=(a.hold, a.hold))
over.update(ROBOT)
for kv in a.set:
    k, v = kv.split("=", 1); x = json.loads(v)
    over[k] = tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x
cfg = env_config(d, **over); env = SlenderBipedEnv(cfg); N = a.n; nst = int(a.T / cfg.ctrl_dt); nj = env.nj
wenv = bw.DomainRandomizationVmapWrapper(env, functools.partial(randomization_fn, rng=jax.random.split(jax.random.PRNGKey(1), N)))
st = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(0), N))
def body(c, k):
    st, dead = c; st2 = wenv.step(st, W(st.obs["state"])); dead = dead | (st2.done > 0); ps = st2.pipeline_state
    return (st2, dead), (ps.qpos[:, 3:7], st2.info["phase"], ps.qpos[:, 7:7 + nj] + st2.info.get("flex", jp.zeros((N, nj))), dead)
(st, dead), (quat, ph, q, dd) = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(nst)))((st, jp.zeros(N, bool)))
quat, ph, q, dd = map(np.asarray, (quat, ph, q, dd))
w, x, y, z = [quat[..., k] for k in range(4)]
G = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
roll = np.degrees(np.arctan2(-G[..., 1], -G[..., 2]))
tab, zz, ev = sc.table(a.run); stn = zz["stance"][-1]; n = len(stn)
m, dm, tid = sc._m, sc._d, sc._tid
def feet(qq, g):
    dm.qpos[:] = 0; dm.qpos[2] = 0.3; dm.qpos[3:7] = sc.gap_fit._quat_from_g(g); dm.qpos[7:] = qq; mujoco.mj_kinematics(m, dm)
    return dm.geom_xpos[tid[0]].copy(), dm.geom_xpos[tid[1]].copy()
k0 = int(round(a.hold / cfg.ctrl_dt)); J = list(env.joints)
iL, iR = J.index("L_hip_roll"), J.index("R_hip_roll")
steps = {i: [] for i in range(4)}; exc = []; fell3 = 0; hr = []
for k in range(N):
    hr.append(np.degrees(q[k0 - 1, k, [iL, iR]]))
    r0 = roll[k0 - 1, k]; seg = slice(k0, min(k0 + 150, nst))
    exc.append(np.abs(roll[seg, k] - r0).max()); fell3 += bool(dd[min(k0 + 150, nst - 1), k])
    cnt = 0
    for i in range(k0, nst - 7):
        if dd[i + 6, k] or cnt >= 4: break
        p0 = int((ph[i, k] / 2 / np.pi % 1) * n) % n; p1 = int((ph[i + 1, k] / 2 / np.pi % 1) * n) % n
        for s_ in (0, 1):
            if stn[p0][s_] < .5 and stn[p1][s_] >= .5 and cnt < 4:
                L, R = feet(q[i + 6, k], G[i + 6, k]); sw, sd = (L, R) if s_ == 0 else (R, L)
                steps[cnt].append(("LR"[s_], (sw[0] - sd[0]) * 1000, abs(L[1] - R[1]) * 1000)); cnt += 1
hr = np.array(hr)
print(f"opening {a.run} ({p.name}) vx {a.vx} hold {a.hold}s n {N} {' '.join(a.set)}")
print(f"  end of hold: hip roll L {np.median(hr[:, 0]):+.1f} R {np.median(hr[:, 1]):+.1f} deg | first 3 s: roll excursion "
      f"p50 {np.median(exc):.1f} p90 {np.percentile(exc, 90):.1f} deg, fallen {fell3}/{N}")
for i in range(4):
    if steps[i]:
        sd = [s[0] for s in steps[i]]; dx = np.array([s[1] for s in steps[i]]); dy = np.array([s[2] for s in steps[i]])
        print(f"  step {i + 1} ({max(set(sd), key=sd.count)}): ahead {np.median(dx):+.0f} mm (p10 {np.percentile(dx, 10):+.0f}), spacing {np.median(dy):.0f} mm")
