"""Round 27: the robot's per-side step stats (limp.py) in the GPU 'sim robot' at a constant command: landing dx / dy,
foot air time vs scheduled, body roll at touchdown and during the swing, hip-roll spread (weight shift) at lift-off.
    scripts/gpu_python.sh scripts/robot27_analysis/limp_sim.py <run> [--vx 0.02] [--nogate] [--set k=json ...]"""
import argparse, functools, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "robot18_analysis"))
import jax, jax.numpy as jp, numpy as np, mujoco
from brax.envs.wrappers import training as bw
from push_eval import ROBOT, load_policy
from slender_biped.mjx_env import SlenderBipedEnv, randomization_fn
from slender_biped.runcfg import env_config
import swing_clear as sc
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--vx", type=float, default=0.02)
ap.add_argument("--T", type=float, default=20.0); ap.add_argument("--n", type=int, default=24); ap.add_argument("--nogate", action="store_true")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args(); W, p, d = load_policy(a.run)
over = dict(step_err=(), reset_stagger=0.0, push_dv=(0.0, 0.0), push_interval_s=(1000.0, 1000.0), push_vel=0.0, act_lpf=0.7,
            cmd_resample_s=(1000.0, 1000.0), reset_hard_prob=0.0, reset_joint_noise=0.02, reset_tilt=0.02,
            cmd_vx=(a.vx, a.vx), cmd_zero_prob=0.0, cmd_wz=(0.0, 0.0), sup=False, turn_prob=0.0)
over.update(ROBOT)
if a.nogate: over["settle_gate_u"] = ()
for kv in a.set:
    k, v = kv.split("=", 1); x = json.loads(v)
    over[k] = tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x
cfg = env_config(d, **over); env = SlenderBipedEnv(cfg); N = a.n; nst = int(a.T / cfg.ctrl_dt); nj = env.nj
wenv = bw.DomainRandomizationVmapWrapper(env, functools.partial(randomization_fn, rng=jax.random.split(jax.random.PRNGKey(1), N)))
st = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(0), N))
st = st.replace(info={**st.info, "phase": jp.zeros_like(st.info["phase"])})
def body(c, k):
    st, dead = c; st2 = wenv.step(st, W(st.obs["state"])); dead = dead | (st2.done > 0); ps = st2.pipeline_state
    return (st2, dead), (ps.qpos[:, 3:7], st2.info["phase"], ps.qpos[:, 7:7 + nj] + st2.info.get("flex", jp.zeros((N, nj))), dead)
(st, dead), (quat, ph, q, dd) = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(nst)))((st, jp.zeros(N, bool)))
quat, ph, q, dd = map(np.asarray, (quat, ph, q, dd))
w, x, y, z = [quat[..., i] for i in range(4)]
G = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
roll = np.degrees(np.arctan2(-G[..., 1], -G[..., 2]))
tab, zz, ev = sc.table(a.run.split(":")[0]); stn = zz["stance"][-1]; n = len(stn)
m, dm, tid = sc._m, sc._d, sc._tid
def feet(qq, g):
    dm.qpos[:] = 0; dm.qpos[2] = 0.3; dm.qpos[3:7] = sc.gap_fit._quat_from_g(g); dm.qpos[7:] = qq; mujoco.mj_kinematics(m, dm)
    return dm.geom_xpos[tid[0]].copy(), dm.geom_xpos[tid[1]].copy()
J = list(env.joints); iL, iR = J.index("L_hip_roll"), J.index("R_hip_roll")
O = {"L": [], "R": []}
for k in range(N):
    r0 = roll[:25, k].mean(); nlift = 0
    for i in range(0, nst - 40):
        if dd[min(i + 40, nst - 1), k]: break
        u0 = ph[i, k] / 2 / np.pi % 1; u1 = ph[i + 1, k] / 2 / np.pi % 1
        p0 = int(u0 * n) % n; p1 = int(u1 * n) % n
        for sd in (0, 1):
            if stn[p0][sd] >= .5 and stn[p1][sd] < .5:
                nlift += 1
                nt = [j for j in range(i + 1, min(i + 80, nst - 7)) if stn[int((ph[j, k] / 2 / np.pi % 1) * n) % n][sd] >= .5]
                if not nt or nlift <= 4: continue
                td = nt[0]; j = td + 6
                FK = [feet(q[t, k], G[t, k]) for t in range(i, min(td + 30, nst))]
                hgt = np.array([(F[sd][2] - F[1 - sd][2]) * 1000 for F in FK]); air = np.where(hgt > 4)[0]
                L, R = feet(q[j, k], G[j, k]); sw, sdd = (L, R) if sd == 0 else (R, L)
                spread = np.degrees(q[i, k, iL] - q[i, k, iR]) * (1 if sd == 0 else -1)
                O["LR"[sd]].append(dict(dx=(sw[0] - sdd[0]) * 1000, dy=abs(L[1] - R[1]) * 1000,
                    t_air=(air[-1] - air[0] + 1) * cfg.ctrl_dt if len(air) else 0, t_sched=(td - i) * cfg.ctrl_dt,
                    roll_td=roll[td, k] - r0, rmin=(roll[i:j, k] - r0).min(), rmax=(roll[i:j, k] - r0).max(), spread=spread,
                    roll_lo=roll[i, k] - r0))
print(f"limp_sim {a.run} ({p.name}) vx {a.vx} {'nogate ' if a.nogate else ''}{' '.join(a.set)}  up {int((~dd[-1]).sum())}/{N}")
for sd in "LR":
    x = O[sd]
    if not x: continue
    md = lambda kk: np.median([e[kk] for e in x])
    print(f"  {sd}: dx {md('dx'):+4.0f} (p10 {np.percentile([e['dx'] for e in x], 10):+4.0f}) dy {md('dy'):3.0f} | air {md('t_air'):.2f}s of {md('t_sched'):.2f} "
          f"| roll at lift-off {md('roll_lo'):+5.1f} at td {md('roll_td'):+5.1f} range {md('rmin'):+5.1f}..{md('rmax'):+5.1f} | shift {md('spread'):+5.1f}")
